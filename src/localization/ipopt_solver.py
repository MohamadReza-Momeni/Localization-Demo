import numpy as np
from .base_solver import BaseSolver


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
        dists = np.linalg.norm(self.anchors - position, axis=1) + 1e-9
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

    # NOTE: hessianstructure/hessian are intentionally unused now —
    # IPOPT is told to approximate the Hessian itself (see hessian_approximation option below).
    # Leaving these as empty stubs is fine ONLY because that option is set.


class IPOPTSolver(BaseSolver):

    def solve(self, anchors, distances, x0=None, ref_power=-40, ple=2.2, weights=None, x_range=(0, 1000),
              y_range=(0, 1000), multi_start=True, verbose=False):
        """
        multi_start=True (default): runs IPOPT from three different starting points —
        the provided warm start (x0), the anchor centroid, and a random point inside
        the bounds — then keeps whichever converges to the LOWEST objective value
        (not just whichever finishes first). This guards against the warm start
        trapping IPOPT in a bad local basin on this non-convex RSSI objective,
        particularly near the edges of the simulation box where GDOP is worst.

        Set multi_start=False to fall back to the old single-start behavior (faster,
        but re-introduces the "IPOPT inherits vanilla's local basin" risk).

        verbose=True prints each candidate's starting point, converged solution, and
        objective value, so you can directly see whether the warm start was actually
        limiting the result or whether all three starts agree (in which case the
        warm start wasn't costing you anything).
        """
        try:
            import cyipopt
        except ImportError as exc:
            raise RuntimeError("cyipopt is not installed in this environment.") from exc

        anchors = np.asarray(anchors)
        distances = np.asarray(distances)

        problem = RSSIDomainProblem(
            anchors, distances, ref_power=ref_power, ple=ple, weights=weights
        )

        # Build the set of starting points to try.
        starting_points = {}
        if x0 is not None:
            starting_points["warm_start"] = np.asarray(x0, dtype=float)
        else:
            # No warm start provided (e.g. this IS the initial vanilla-equivalent call) —
            # centroid doubles as both the "no info" default and one of the multi-start points.
            starting_points["warm_start"] = np.mean(anchors, axis=0)

        starting_points["anchor_centroid"] = np.mean(anchors, axis=0)

        rng = np.random.default_rng()
        starting_points["random_point"] = np.array([
            rng.uniform(x_range[0], x_range[1]),
            rng.uniform(y_range[0], y_range[1]),
        ])

        if not multi_start:
            starting_points = {"warm_start": starting_points["warm_start"]}

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
            nlp.add_option("sb", "yes")  # suppress startup banner
            nlp.add_option("print_level", 0)
            nlp.add_option("max_iter", 500)
            nlp.add_option("tol", 1e-6)
            nlp.add_option("hessian_approximation", "limited-memory")

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

        best = min(candidates, key=lambda c: c["objective"])

        if verbose:
            print(f"  [ipopt multi-start] BEST: {best['label']} (objective={best['objective']:.4f})")

        return {
            "solution": best["solution"],
            "success": best["status"] in [0, 1],
            "info": best["info"],
            "chosen_start": best["label"],
            "multi_start_candidates": candidates,
        }