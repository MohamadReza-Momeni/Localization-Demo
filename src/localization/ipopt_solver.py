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
              y_range=(0, 1000)):
        try:
            import cyipopt
        except ImportError as exc:
            raise RuntimeError("cyipopt is not installed in this environment.") from exc

        anchors = np.asarray(anchors)
        distances = np.asarray(distances)

        if x0 is None:
            x0 = np.mean(anchors, axis=0)

        problem = RSSIDomainProblem(
            anchors, distances, ref_power=ref_power, ple=ple, weights=weights
        )

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
        nlp.add_option("hessian_approximation", "limited-memory")  # <-- the actual fix

        x_opt, info = nlp.solve(x0)

        return {
            "solution": x_opt,
            "success": info["status"] in [0, 1],
            "info": info,
        }