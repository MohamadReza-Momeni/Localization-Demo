import numpy as np
from scipy.optimize import least_squares
from .base_solver import BaseSolver


class SciPyLocalizationSolver(BaseSolver):

    def solve(self, anchors, distances, x0=None, weights=None):
        anchors = np.asarray(anchors)
        distances = np.asarray(distances)

        if x0 is None:
            x0 = np.mean(anchors, axis=0)

        if weights is None:
            weights = np.ones(len(anchors))
        else:
            weights = np.asarray(weights)

        result = least_squares(
            self._residuals,
            x0,
            args=(anchors, distances, weights),
            method="lm",
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