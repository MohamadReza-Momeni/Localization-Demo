import numpy as np

from .base_solver import BaseSolver


class RSSILocalizationProblem:
    def __init__(self, anchors, distances):
        self.anchors = np.asarray(anchors)
        self.distances = np.asarray(distances)

    def objective(self, position):
        residuals = self._residuals(position)
        return np.sum(residuals**2)

    def gradient(self, position):
        px, py = position
        gradient = np.zeros(2)

        for anchor, distance in zip(self.anchors, self.distances):
            dx = px - anchor[0]
            dy = py - anchor[1]
            estimated_distance = np.sqrt(dx**2 + dy**2) + 1e-9
            residual = estimated_distance - distance

            gradient += 2 * residual * np.array([
                dx / estimated_distance,
                dy / estimated_distance,
            ])

        return gradient

    def constraints(self, position):
        return np.array([])

    def jacobian(self, position):
        return np.array([])

    def hessianstructure(self):
        return np.array([])

    def hessian(self, position, lagrange, obj_factor):
        return np.array([])

    def _residuals(self, position):
        estimated_distances = np.linalg.norm(self.anchors - position, axis=1)
        return estimated_distances - self.distances


class IPOPTSolver(BaseSolver):
    def solve(self, anchors, distances, x0=None):
        try:
            import ipopt
        except ImportError as exc:
            raise RuntimeError("IPOPT is not installed in this environment.") from exc

        anchors = np.asarray(anchors)
        distances = np.asarray(distances)

        if x0 is None:
            x0 = np.mean(anchors, axis=0)

        problem = RSSILocalizationProblem(anchors, distances)
        nlp = ipopt.problem(
            n=2,
            m=0,
            problem_obj=problem,
            lb=[-1e6, -1e6],
            ub=[1e6, 1e6],
        )

        nlp.addOption("print_level", 0)
        nlp.addOption("max_iter", 100)

        solution, info = nlp.solve(x0)

        return {
            "solution": solution,
            "info": info,
        }
