import numpy as np
from .base_solver import BaseSolver
from .ipopt_params import IPOPTHyperparams
from .multistart import generate_starting_points


class RSSIDomainProblem:
    def __init__(self, anchors, distances, ref_power=-40, ple=2.2, weights=None):
        self.anchors = np.asarray(anchors)
        self.rssi_meas = ref_power - 10.0 * ple * np.log10(np.asarray(distances) + 1e-9)
        self.ref_power = ref_power
        self.ple = ple
        self.log_factor = 10.0 * self.ple / np.log(10.0)

        if weights is None:
            self.weights = np.ones(len(anchors))
        else:
            self.weights = np.asarray(weights)

    def objective(self, position):
        # NOTE: uses sqrt(sq_dist + eps), not norm(...) + eps, so this is
        # EXACTLY the function gradient() differentiates (same eps convention
        # as d2/d there) — keeps the analytic gradient consistent with the
        # coded objective even in the degenerate case where the iterate lands
        # on top of an anchor (previously these used two different epsilon
        # conventions that only disagreed at that single point).
        sq_dists = np.sum((self.anchors - position) ** 2, axis=1) + 1e-9
        dists = np.sqrt(sq_dists)
        rssi_pred = self.ref_power - 10.0 * self.ple * np.log10(dists)
        residuals = self.rssi_meas - rssi_pred
        return np.sum(self.weights * (residuals ** 2))

    def gradient(self, position):
        px, py = position
        gradient = np.zeros(2)

        for anchor, r_meas, w in zip(self.anchors, self.rssi_meas, self.weights):
            dx = px - anchor[0]
            dy = py - anchor[1]
            d2 = dx ** 2 + dy ** 2 + 1e-9
            d = np.sqrt(d2)

            r_pred = self.ref_power - 10.0 * self.ple * np.log10(d)
            residual = r_meas - r_pred
            grad_factor = self.log_factor * (1.0 / d2)

            gradient[0] += 2.0 * w * residual * grad_factor * dx
            gradient[1] += 2.0 * w * residual * grad_factor * dy

        return gradient

    def constraints(self, position):
        return np.array([])

    def jacobian(self, position):
        return np.array([])

    # --- Exact Hessian of the objective ---
    # Enables hessian_approximation="exact" (previously these were empty stubs,
    # so requesting "exact" crashed with "Hessian callback not defined").
    #
    # Objective: f(theta) = sum_i w_i * r_i^2, with
    #   r_i    = rssi_meas_i - (P0 - 10*ple*log10(d_i)) = rssi_meas_i - P0 + 10*ple*log10(d_i)
    #   u_i    = theta - anchor_i,   d2_i = u_i . u_i
    #   dr/dtheta = k * u_i / d2_i           (k = log_factor = 10*ple/ln(10))
    #
    # Per-anchor Hessian block (Gauss-Newton term + curvature term):
    #   2*w_i * [ k^2 * (u u^T)/d2^2  +  r_i * k * ( I/d2 - 2 (u u^T)/d2^2 ) ]
    #
    # cyipopt wants only the LOWER triangle, in the order given by
    # hessianstructure(), scaled by obj_factor (there are no constraints, m=0,
    # so the lambda term vanishes).
    def hessianstructure(self):
        # Lower triangle of a 2x2: (0,0), (1,0), (1,1).
        return (np.array([0, 1, 1]), np.array([0, 0, 1]))

    def hessian(self, position, lagrange, obj_factor):
        px, py = position
        k = self.log_factor
        H = np.zeros((2, 2))

        for anchor, r_meas, w in zip(self.anchors, self.rssi_meas, self.weights):
            ux = px - anchor[0]
            uy = py - anchor[1]
            d2 = ux ** 2 + uy ** 2 + 1e-9
            d = np.sqrt(d2)

            r_pred = self.ref_power - 10.0 * self.ple * np.log10(d)
            residual = r_meas - r_pred

            # outer product u u^T
            uxx = ux * ux
            uxy = ux * uy
            uyy = uy * uy

            gn = (k * k) / (d2 * d2)              # Gauss-Newton coefficient on u u^T
            curv = residual * k                    # curvature coefficient
            inv_d2 = 1.0 / d2
            two_over_d4 = 2.0 / (d2 * d2)

            # 2*w * [ gn*uuT + curv*( I/d2 - two_over_d4*uuT ) ]
            H[0, 0] += 2.0 * w * (gn * uxx + curv * (inv_d2 - two_over_d4 * uxx))
            H[1, 0] += 2.0 * w * (gn * uxy + curv * (-two_over_d4 * uxy))
            H[1, 1] += 2.0 * w * (gn * uyy + curv * (inv_d2 - two_over_d4 * uyy))

        H *= obj_factor
        # Return lower triangle in hessianstructure() order: (0,0),(1,0),(1,1).
        return np.array([H[0, 0], H[1, 0], H[1, 1]])


class IPOPTSolver(BaseSolver):

    def solve(self, anchors, distances, x0=None, ref_power=-40, ple=2.2, weights=None,
              x_range=(0, 1000), y_range=(0, 1000),
              hyperparams: IPOPTHyperparams = None, verbose=False):
        """
        hyperparams (IPOPTHyperparams): every IPOPT-internal knob — tolerances,
        max_iter, hessian_approximation, mu_strategy, which starting points feed
        multi-start, and whether multi-start runs at all — bundled as a single
        explicit, loggable, sweepable project parameter. Defaults to
        IPOPTHyperparams() (the old hardcoded behavior) if not provided.

        Which starting points actually run is controlled by
        hyperparams.starting_points and hyperparams.multi_start:
          - "warm_start": the provided x0 (or hyperparams.fixed_initial_point,
            which overrides x0 if set — e.g. to reproduce a MATLAB run's init).
          - "anchor_centroid": mean of anchor positions.
          - "random_point": uniform random point inside [x_range] x [y_range].
        Whichever converges to the LOWEST objective value is kept — this guards
        against the warm start trapping IPOPT in a bad local basin on this
        non-convex RSSI objective, particularly near the edges of the box
        where GDOP is worst.

        verbose=True prints each candidate's starting point, converged solution,
        and objective value.
        """
        try:
            import cyipopt
        except ImportError as exc:
            raise RuntimeError("cyipopt is not installed in this environment.") from exc

        if hyperparams is None:
            hyperparams = IPOPTHyperparams()

        anchors = np.asarray(anchors)
        distances = np.asarray(distances)

        problem = RSSIDomainProblem(
            anchors, distances, ref_power=ref_power, ple=ple, weights=weights
        )

        # Build the set of starting points to try, restricted to
        # hyperparams.starting_points (order preserved for reproducibility).
        # Use the SHARED generator (multistart.py) so native cyipopt and the
        # AMPL-routed solvers start from byte-identical points — including a
        # scenario-seeded, reproducible "random_point" — which is what makes
        # the cross-solver comparison isolate the solver, not the RNG.
        # `fixed_initial_point` still overrides the warm start when set.
        warm_x0 = hyperparams.fixed_initial_point if hyperparams.fixed_initial_point is not None else x0
        all_points = generate_starting_points(anchors, warm_x0, x_range, y_range)

        if hyperparams.multi_start:
            labels_to_run = hyperparams.starting_points
        else:
            labels_to_run = (hyperparams.starting_points[0],)

        starting_points = {label: all_points[label] for label in labels_to_run}

        candidates = []
        for label, start in starting_points.items():
            start_clipped = np.array([
                np.clip(start[0], x_range[0], x_range[1]),
                np.clip(start[1], y_range[0], y_range[1]),
            ])

            nlp = cyipopt.Problem(
                n=2,
                m=0,
                problem_obj=problem,
                lb=np.array([x_range[0], y_range[0]]),
                ub=np.array([x_range[1], y_range[1]]),
                cl=np.array([]),
                cu=np.array([])
            )
            for option_name, option_value in hyperparams.as_ipopt_options().items():
                nlp.add_option(option_name, option_value)

            x_opt, info = nlp.solve(start_clipped)
            objective_val = problem.objective(x_opt)

            candidates.append({
                "label": label,
                "start": start_clipped,
                "solution": x_opt,
                "objective": objective_val,
                "status": info["status"],
                "info": info,
            })

            if verbose:
                print(
                    f"  [ipopt multi-start] {label:<15} "
                    f"start=({start_clipped[0]:7.2f},{start_clipped[1]:7.2f})  "
                    f"-> solution=({x_opt[0]:7.2f},{x_opt[1]:7.2f})  "
                    f"objective={objective_val:.4f}  status={info['status']}"
                )

        # Pick the best candidate. Two criteria:
        #   1. prefer a CONVERGED candidate (status in {0,1}) over a
        #      non-converged one, because a non-converged run (e.g. IPOPT hit
        #      max_iter) may have stopped short and we don't want to report it
        #      as the trusted answer / mark it success=True;
        #   2. among candidates that tie on convergence, keep the LOWEST
        #      objective — this is the right tie-breaker on the non-convex RSSI
        #      objective, where the lowest-objective point is the intended
        #      estimate even if another start also "converged" slightly higher.
        # Fall back to the lowest-objective non-converged candidate only if
        # every start failed to converge.
        converged = [c for c in candidates if c["status"] in (0, 1)]
        pool = converged if converged else candidates
        best = min(pool, key=lambda c: c["objective"])
        # success reflects the ACTUAL returned candidate's convergence, so the
        # reported success flag is honest (see PROJECT_NOTES.md audit section).
        best_converged = bool(best["status"] in (0, 1))

        if verbose:
            print(f"  [ipopt multi-start] BEST: {best['label']} (objective={best['objective']:.4f}, "
                  f"status={best['status']}, converged={best_converged})")

        return {
            "solution": best["solution"],
            "success": best_converged,
            "info": best["info"],
            "chosen_start": best["label"],
            "objective": best["objective"],
            "multi_start_candidates": candidates,
            "hyperparams": hyperparams,
        }
