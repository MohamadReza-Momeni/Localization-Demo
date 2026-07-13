import numpy as np
from scipy.optimize import least_squares
from .base_solver import BaseSolver


class SciPyLocalizationSolver(BaseSolver):

    def solve(self, anchors, distances, x0=None, weights=None, x_range=(0, 1000), y_range=(0, 1000)):
        anchors = np.asarray(anchors)
        distances = np.asarray(distances)

        if x0 is None:
            x0 = np.mean(anchors, axis=0)
        # Clip x0 into bounds in case a caller passes an out-of-box warm start —
        # "trf" requires the initial guess to already satisfy the bounds.
        x0 = np.array([
            np.clip(x0[0], x_range[0], x_range[1]),
            np.clip(x0[1], y_range[0], y_range[1]),
        ])

        if weights is None:
            weights = np.ones(len(anchors))
        else:
            weights = np.asarray(weights)

        # UPDATED: "lm" (Levenberg-Marquardt) does not support bounds at all, so this
        # solver could previously report positions outside the simulation area (e.g.
        # negative coordinates) even though the true target is always generated inside
        # x_range/y_range. Switched to "trf" (Trust Region Reflective), which supports
        # bounds and gives ipopt/particle_filter-consistent, physically valid answers.
        result = least_squares(
            self._residuals,
            x0,
            args=(anchors, distances, weights),
            method="trf",
            bounds=([x_range[0], y_range[0]], [x_range[1], y_range[1]]),
        )

        return {
            "solution": result.x,
            "cost": result.cost,
            "success": result.success,
            "message": result.message,
            "iterations": result.nfev,
        }

    def _residuals(self, position, anchors, distances, weights):
        estimated = np.linalg.norm(anchors - position, axis=1)
        residuals = estimated - distances

        if weights is None:
            return residuals

        return np.sqrt(weights) * residuals