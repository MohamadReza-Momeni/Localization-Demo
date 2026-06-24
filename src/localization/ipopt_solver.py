import numpy as np


class RSSILocalizationProblem:
    """
    IPOPT formulation for RSSI-based localization (2D).
    """

    def __init__(self, anchors, distances):
        self.anchors = np.asarray(anchors)
        self.distances = np.asarray(distances)

        self.n_anchors = len(anchors)

    # -------------------------
    # Objective function
    # -------------------------
    def objective(self, x):
        px, py = x[0], x[1]

        error = 0.0
        for i in range(self.n_anchors):
            ax, ay = self.anchors[i]
            d_hat = np.sqrt((px - ax)**2 + (py - ay)**2)
            error += (d_hat - self.distances[i])**2

        return error

    # -------------------------
    # Gradient (IMPORTANT for IPOPT)
    # -------------------------
    def gradient(self, x):
        px, py = x[0], x[1]

        grad_x = 0.0
        grad_y = 0.0

        for i in range(self.n_anchors):
            ax, ay = self.anchors[i]

            dx = px - ax
            dy = py - ay

            dist = np.sqrt(dx**2 + dy**2) + 1e-9  # avoid division by zero

            diff = dist - self.distances[i]

            grad_x += 2 * diff * (dx / dist)
            grad_y += 2 * diff * (dy / dist)

        return np.array([grad_x, grad_y])

    # -------------------------
    # Required by IPOPT
    # -------------------------
    def constraints(self, x):
        return np.array([])

    def jacobian(self, x):
        return np.array([])

    def hessianstructure(self):
        return np.array([])

    def hessian(self, x, lagrange, obj_factor):
        return np.array([])

class IPOPTSolver:
    def __init__(self, anchors, distances):
        self.problem = RSSILocalizationProblem(anchors, distances)

    def solve(self, x0=None):
        if x0 is None:
            x0 = np.mean(self.problem.anchors, axis=0)

        try:
            import ipopt
        except ImportError:
            return self._solve_with_gradient_descent(x0)

        nlp = ipopt.problem(
            n=2,
            m=0,
            problem_obj=self.problem,
            lb=[-1e6, -1e6],
            ub=[1e6, 1e6],
        )

        nlp.addOption("print_level", 0)
        nlp.addOption("max_iter", 100)

        x, info = nlp.solve(x0)

        return {
            "solution": x,
            "info": {
                "solver": "ipopt",
                **info,
            }
        }

    def _solve_with_gradient_descent(self, x0, max_iter=5000, tolerance=1e-8):
        x = np.asarray(x0, dtype=float)
        objective = self.problem.objective(x)
        step = 1.0

        for iteration in range(max_iter):
            grad = self.problem.gradient(x)
            grad_norm = np.linalg.norm(grad)

            if grad_norm < tolerance:
                break

            accepted = False
            trial_step = step

            for _ in range(30):
                candidate = x - trial_step * grad
                candidate_objective = self.problem.objective(candidate)

                if candidate_objective < objective:
                    x = candidate
                    objective = candidate_objective
                    step = min(trial_step * 1.2, 10.0)
                    accepted = True
                    break

                trial_step *= 0.5

            if not accepted:
                break

        return {
            "solution": x,
            "info": {
                "solver": "gradient_descent",
                "status": "ipopt_not_installed",
                "iterations": iteration + 1,
                "objective": objective,
                "gradient_norm": np.linalg.norm(self.problem.gradient(x)),
            }
        }
