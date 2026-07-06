import numpy as np
from .base_solver import BaseSolver


class RSSIDomainProblem:

    # 1. Add weights to the initializer
    def __init__(self, anchors, distances, ref_power=-40, ple=2.2, weights=None):
        self.anchors = np.asarray(anchors)
        self.rssi_meas = ref_power - 10.0 * ple * np.log10(np.asarray(distances) + 1e-9)
        self.ref_power = ref_power
        self.ple = ple
        self.log_factor = 10.0 * self.ple / np.log(10.0)
        
        # Default to unweighted (1.0) if no weights are provided
        if weights is None:
            self.weights = np.ones(len(anchors))
        else:
            self.weights = np.asarray(weights)

    def objective(self, position):
        dists = np.linalg.norm(self.anchors - position, axis=1) + 1e-9
        rssi_pred = self.ref_power - 10.0 * self.ple * np.log10(dists)

        residuals = self.rssi_meas - rssi_pred
        
        # 2. Multiply the squared residuals by the weights
        return np.sum(self.weights * (residuals ** 2))

    def gradient(self, position):
        px, py = position
        gradient = np.zeros(2)

        # 3. Zip the weights into the loop to apply them to the gradient
        for anchor, r_meas, w in zip(self.anchors, self.rssi_meas, self.weights):
            dx = px - anchor[0]
            dy = py - anchor[1]
            d2 = dx ** 2 + dy ** 2 + 1e-9
            d = np.sqrt(d2)

            r_pred = self.ref_power - 10.0 * self.ple * np.log10(d)
            residual = r_meas - r_pred
            
            # The chain rule derivative factor
            grad_factor = self.log_factor * (1.0 / d2)
            
            # 4. Multiply the gradient step by the specific anchor's weight (w)
            gradient[0] += 2.0 * w * residual * grad_factor * dx
            gradient[1] += 2.0 * w * residual * grad_factor * dy

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