from src.scenario.point_generator import PointGenerator


class AnchorGenerator(PointGenerator):
    def __init__(self, count, x_range, y_range, seed=42):
        super().__init__(count, x_range, y_range, seed)
