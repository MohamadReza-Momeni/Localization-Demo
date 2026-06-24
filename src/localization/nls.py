import numpy as np

class NLSProblem:
    def __init__(self, anchors, distances):
        self.anchors = anchors
        self.distances = distances

    def objective(self, vars):
        x, y = vars
        err = 0.0

        for i, a in enumerate(self.anchors):
            d_hat = np.linalg.norm(np.array([x, y]) - a)
            err += (d_hat - self.distances[i]) ** 2

        return err