import numpy as np

class TargetGenerator:
    def __init__(self, t, x_range, y_range, seed=1):
        self.t = t
        self.rng = np.random.default_rng(seed)
        self.x_range = x_range
        self.y_range = y_range

    def generate(self):
        x = self.rng.uniform(*self.x_range, self.t)
        y = self.rng.uniform(*self.y_range, self.t)
        return np.column_stack([x, y])