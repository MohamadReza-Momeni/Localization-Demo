from dataclasses import dataclass, field
import numpy as np
from src.localization.ipopt_params import IPOPTHyperparams


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
    ipopt_params: IPOPTHyperparams = field(default_factory=IPOPTHyperparams)
    # Per-solver-name raw AMPL option strings, e.g. {"bonmin": "bonmin.algorithm=B-BB"}.
    # See ampl_solver.py's AMPLSolver.solve docstring for why this is a raw
    # passthrough rather than a typed dataclass like IPOPTHyperparams.
    ampl_options: dict = field(default_factory=dict)
