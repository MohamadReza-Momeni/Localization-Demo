import numpy as np


class PointGenerator:
    def __init__(self, count, x_range, y_range, seed):
        self.count = count
        self.x_range = x_range
        self.y_range = y_range
        self.rng = np.random.default_rng(seed)

    def generate(self):
        x_coordinates = self.rng.uniform(self.x_range[0], self.x_range[1], self.count)
        y_coordinates = self.rng.uniform(self.y_range[0], self.y_range[1], self.count)
        return np.column_stack([x_coordinates, y_coordinates])
