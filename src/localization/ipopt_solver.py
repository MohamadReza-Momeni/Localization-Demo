import numpy as np
from .base_solver import BaseSolver


class RSSIDomainProblem:

    def __init__(self, anchors, distances, ref_power=-40, ple=2.2):
        self.anchors = np.asarray(anchors)

        # Reverse-engineer raw RSSI values from distance estimates
        # to preserve the existing repository pipeline structure safely
        self.rssi_meas = ref_power - 10.0 * ple * np.log10(np.asarray(distances) + 1e-9)

        self.ref_power = ref_power
        self.ple = ple
        self.log_factor = 10.0 * self.ple / np.log(10.0)

    def objective(self, position):
        dists = np.linalg.norm(self.anchors - position, axis=1) + 1e-9
        rssi_pred = self.ref_power - 10.0 * self.ple * np.log10(dists)

        # Optimize inside the symmetric Gaussian RSSI decibel space
        residuals = self.rssi_meas - rssi_pred
        return np.sum(residuals ** 2)

    def gradient(self, position):
        px, py = position
        gradient = np.zeros(2)

        for anchor, r_meas in zip(self.anchors, self.rssi_meas):
            dx = px - anchor[0]
            dy = py - anchor[1]
            d2 = dx ** 2 + dy ** 2 + 1e-9
            d = np.sqrt(d2)

            r_pred = self.ref_power - 10.0 * self.ple * np.log10(d)
            residual = r_meas - r_pred

            dr_dx = (self.log_factor * dx) / d2
            dr_dy = (self.log_factor * dy) / d2

            gradient[0] += 2.0 * residual * dr_dx
            gradient[1] += 2.0 * residual * dr_dy

        return gradient

    def constraints(self, position):
        return np.array([])

    def jacobian(self, position):
        return np.array([])

    def hessianstructure(self):
        # FIXED: Returns a valid Cython coordinate tuple to satisfy C-extension initializations
        return (np.array([], dtype=np.int32), np.array([], dtype=np.int32))

    def hessian(self, position, lagrange, obj_factor):
        return np.array([])


class IPOPTSolver(BaseSolver):

    def solve(self, anchors, distances, x0=None, ref_power=-40, ple=2.2):
        try:
            import cyipopt
        except ImportError as exc:
            raise RuntimeError(
                "cyipopt is not installed in this environment."
            ) from exc

        anchors = np.asarray(anchors)
        distances = np.asarray(distances)

        if x0 is None:
            x0 = np.mean(anchors, axis=0)

        problem = RSSIDomainProblem(anchors, distances, ref_power=ref_power, ple=ple)

        nlp = cyipopt.Problem(
            n=2,
            m=0,
            problem_obj=problem,
            lb=np.array([0.0, 0.0]),  # Keep target inside tracking grid
            ub=np.array([1000.0, 1000.0]),  # Keep target inside tracking grid
            cl=np.array([]),
            cu=np.array([])
        )

        nlp.add_option("print_level", 0)
        nlp.add_option("max_iter", 100)
        nlp.add_option("hessian_approximation", "limited-memory")

        solution, info = nlp.solve(x0)

        return {
            "solution": solution,
            "info": info,
        }