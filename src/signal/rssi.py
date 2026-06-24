import numpy as np

class RSSIModel:
    def __init__(self, P0=-40, n=2.0, sigma=2.0, d0=1.0):
        self.P0 = P0
        self.n = n
        self.sigma = sigma
        self.d0 = d0

    def rssi(self, d):
        noise = np.random.normal(0, self.sigma)
        return self.P0 - 10 * self.n * np.log10(d / self.d0) + noise

    def rssi_matrix(self, anchors, targets):
        R = np.zeros((len(anchors), len(targets)))

        for i, a in enumerate(anchors):
            for j, t in enumerate(targets):
                d = np.linalg.norm(a - t)
                R[i, j] = self.rssi(d)

        return R