import numpy as np
from .base_solver import BaseSolver


class RSSIDomainProblem:

    def __init__(self, anchors, distances, ref_power=-40, ple=2.2, weights=None):
        self.anchors = np.asarray(anchors)
        self.measured_distances = np.asarray(distances, dtype=float)
        self.ref_power = ref_power
        self.ple = ple

        if weights is None:
            self.weights = np.ones(len(anchors))
        else:
            self.weights = np.asarray(weights, dtype=float)

    def objective(self, position):
        predicted_distances = np.linalg.norm(self.anchors - position, axis=1) + 1e-9
        residuals = self.measured_distances - predicted_distances
        return np.sum(self.weights * (residuals ** 2))

    def gradient(self, position):
        px, py = position
        gradient = np.zeros(2)

        for anchor, dist_meas, w in zip(self.anchors, self.measured_distances, self.weights):
            dx = px - anchor[0]
            dy = py - anchor[1]
            d = np.sqrt(dx ** 2 + dy ** 2 + 1e-9)

            residual = dist_meas - d
            gradient[0] += -2.0 * w * residual * dx / d
            gradient[1] += -2.0 * w * residual * dy / d

        return gradient

    def constraints(self, position):
        return np.array([])

    def jacobian(self, position):
        return np.array([])

    def hessianstructure(self):
        return (np.array([], dtype=np.int32), np.array([], dtype=np.int32))

    def hessian(self, position, lagrange, obj_factor):
        return np.array([])


class IPOPTSolver(BaseSolver):

    # 5. Add weights parameter to the solve method
    def solve(self, anchors, distances, x0=None, ref_power=-40, ple=2.2, weights=None, x_range=(0,1000), y_range=(0,1000)):
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
            # UPDATED: Use dynamic bounds for Lower Bound (lb) and Upper Bound (ub)
            lb=np.array([x_range[0], y_range[0]]), 
            ub=np.array([x_range[1], y_range[1]]), 
            cl=np.array([]),
            cu=np.array([])
        )

        # nlp.add_option("print_level", 0)
        nlp.add_option("print_level", 0)
        nlp.add_option("max_iter", 500)
        nlp.add_option("tol", 1e-6)

        x_opt, info = nlp.solve(x0)

        return {
            "solution": x_opt,
            "success": info["status"] in [0, 1],
            "info": info,
        }