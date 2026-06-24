import numpy as np


class TargetGenerator:
    def __init__(self, t, x_range, y_range, seed=1):
        self.t = t
        self.x_range = x_range
        self.y_range = y_range
        self.rng = np.random.default_rng(seed)

    def generate(self):
        x = self.rng.uniform(self.x_range[0], self.x_range[1], self.t)
        y = self.rng.uniform(self.y_range[0], self.y_range[1], self.t)
        return np.column_stack([x, y])