import numpy as np


class AnchorGenerator:
    def __init__(self, n, x_range, y_range, seed=42):
        self.n = n
        self.x_range = x_range
        self.y_range = y_range
        self.rng = np.random.default_rng(seed)

    def generate(self):
        x = self.rng.uniform(self.x_range[0], self.x_range[1], self.n)
        y = self.rng.uniform(self.y_range[0], self.y_range[1], self.n)
        return np.column_stack([x, y])