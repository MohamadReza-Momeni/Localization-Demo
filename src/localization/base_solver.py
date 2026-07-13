from abc import ABC, abstractmethod


class BaseSolver(ABC):
    @abstractmethod
    def solve(self, anchors, distances, x0=None):
        raise NotImplementedError