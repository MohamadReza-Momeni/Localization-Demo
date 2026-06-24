from src.scenario.point_generator import PointGenerator


class TargetGenerator(PointGenerator):
    def __init__(self, count, x_range, y_range, seed=1):
        super().__init__(count, x_range, y_range, seed)
