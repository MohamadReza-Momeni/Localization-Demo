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

    def hessianstructure(self):
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

            uxx = ux * ux
            uxy = ux * uy
            uyy = uy * uy

            gn = (k * k) / (d2 * d2)              
            curv = residual * k                    
            inv_d2 = 1.0 / d2
            two_over_d4 = 2.0 / (d2 * d2)

            H[0, 0] += 2.0 * w * (gn * uxx + curv * (inv_d2 - two_over_d4 * uxx))
            H[1, 0] += 2.0 * w * (gn * uxy + curv * (-two_over_d4 * uxy))
            H[1, 1] += 2.0 * w * (gn * uyy + curv * (inv_d2 - two_over_d4 * uyy))

        H *= obj_factor
        return np.array([H[0, 0], H[1, 0], H[1, 1]])

class IPOPTSolver(BaseSolver):
    def solve(self, anchors, distances, x0=None, ref_power=-40, ple=2.2, weights=None,
              x_range=(0, 1000), y_range=(0, 1000), hyperparams=None, verbose=False):
        try:
            import cyipopt
        except ImportError as exc:
            raise RuntimeError("cyipopt is not installed in this environment.") from exc

        if hyperparams is None:
            hyperparams = IPOPTHyperparams()

        anchors = np.asarray(anchors)
        distances = np.asarray(distances)
        problem = RSSIDomainProblem(anchors, distances, ref_power=ref_power, ple=ple, weights=weights)

        warm_x0 = hyperparams.fixed_initial_point if hyperparams.fixed_initial_point is not None else x0
        all_points = generate_starting_points(anchors, warm_x0, x_range, y_range)

        labels_to_run = hyperparams.starting_points if hyperparams.multi_start else (hyperparams.starting_points[0],)
        starting_points = {label: all_points[label] for label in labels_to_run}

        candidates = []
        for label, start in starting_points.items():
            start_clipped = np.array([np.clip(start[0], x_range[0], x_range[1]), np.clip(start[1], y_range[0], y_range[1])])

            nlp = cyipopt.Problem(
                n=2, m=0, problem_obj=problem,
                lb=np.array([x_range[0], y_range[0]]), ub=np.array([x_range[1], y_range[1]]),
                cl=np.array([]), cu=np.array([])
            )
            for option_name, option_value in hyperparams.as_ipopt_options().items():
                nlp.add_option(option_name, option_value)

            x_opt, info = nlp.solve(start_clipped)
            objective_val = problem.objective(x_opt)

            candidates.append({
                "label": label, "start": start_clipped, "solution": x_opt,
                "objective": objective_val, "status": info["status"], "info": info
            })

        converged = [c for c in candidates if c["status"] in (0, 1)]
        pool = converged if converged else candidates
        best = min(pool, key=lambda c: c["objective"])
        
        return {
            "solution": best["solution"],
            "success": bool(best["status"] in (0, 1)),
            "info": best["info"],
            "chosen_start": best["label"],
            "objective": best["objective"]
        }