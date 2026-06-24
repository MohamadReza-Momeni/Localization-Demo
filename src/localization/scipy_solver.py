import numpy as np
from scipy.optimize import least_squares
from .base_solver import BaseSolver


class SciPyLocalizationSolver(BaseSolver):

    def __init__(self):
        pass

    def _residuals(self, x, anchors, distances):
        px, py = x
        residuals = []

        for i, a in enumerate(anchors):
            ax, ay = a
            d = np.sqrt((px - ax)**2 + (py - ay)**2)
            residuals.append(d - distances[i])

        return np.array(residuals)

    def solve(self, anchors, distances, x0=None):

        anchors = np.asarray(anchors)
        distances = np.asarray(distances)

        if x0 is None:
            x0 = np.mean(anchors, axis=0)

        result = least_squares(
            self._residuals,
            x0,
            args=(anchors, distances),
            method="lm"   # Levenberg–Marquardt (great for NLS)
        )

        return {
            "solution": result.x,
            "cost": result.cost,
            "success": result.success,
            "message": result.message,
            "iterations": result.nfev
        }