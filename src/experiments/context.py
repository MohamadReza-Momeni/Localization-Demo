from dataclasses import dataclass
import numpy as np

@dataclass
class RunContext:
    """Encapsulates all the data needed by any solver for a single run."""
    anchors: np.ndarray
    distances: np.ndarray
    baseline_guess: np.ndarray
    p0: float
    ple: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]