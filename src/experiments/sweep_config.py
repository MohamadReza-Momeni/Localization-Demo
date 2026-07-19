"""
src/experiments/sweep_config.py

Implements the data-generation loop from the professor's notes:

    "Anchor (generate) -> same for one target -> next P0 -> next B(eta) -> ..."
    "Data, sigmas per P0"

This is structurally different from ExperimentConfig's random sampling
(task.py draws one random P0/beta/sigma per run). Here P0, beta (PLE), and
sigma are each an explicit, ordered grid of values. For a FIXED anchor
layout and target (one "replication"), every (P0, beta, sigma) combination
in the grid is evaluated — so you can isolate how error varies as P0 alone
changes, holding geometry fixed, then do the same for beta, then sigma.
Multiple replications (independent anchor/target draws) give you the
Monte Carlo repeats needed for confidence bounds and, later, CRLB
comparison at each grid point.
"""
from dataclasses import dataclass, field
import numpy as np
from src.localization.ipopt_params import IPOPTHyperparams


@dataclass(frozen=True)
class SweepConfig:
    anchor_count: int = 6
    target_count: int = 1
    x_range: tuple[float, float] = (0, 1000)
    y_range: tuple[float, float] = (0, 1000)

    lat0: float = 35.7152
    lon0: float = 51.4043

    # Explicit grids, in place of ExperimentConfig's *_range random sampling.
    p0_values: tuple[float, ...] = (-40.0,)
    ple_values: tuple[float, ...] = (2.0, 3.0, 4.0)
    sigma_values: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)

    # Heterogeneous (distance-dependent) noise; het_factor=0.0 -> homogeneous
    # (unchanged). See src/signal/rssi.py and ExperimentConfig for details.
    het_factor: float = 0.0
    het_reference_distance: float = 100.0

    # Master RNG seed. None -> OS entropy per replication; int -> reproducible.
    base_seed: int | None = None

    # Independent anchor/target draws per grid sweep — this is the Monte
    # Carlo repeat count. Each replication re-runs the FULL P0 x beta x
    # sigma grid on a fresh anchor layout + target position.
    replications: int = 10

    solvers: tuple[str, ...] = ("vanilla", "weighted", "ipopt", "weighted_ipopt", "particle_filter")
    ipopt_params: IPOPTHyperparams = field(default_factory=IPOPTHyperparams)
    ampl_options: dict = field(default_factory=dict)

    def grid_size(self) -> int:
        """Total (P0, beta, sigma) combinations per target per replication."""
        return len(self.p0_values) * len(self.ple_values) * len(self.sigma_values)

    def total_solver_calls(self) -> int:
        """Rough cost estimate — useful before kicking off a big sweep."""
        return self.replications * self.target_count * self.grid_size() * len(self.solvers)

    @staticmethod
    def make_grid(min_val: float, max_val: float, n: int) -> tuple[float, ...]:
        """Convenience: evenly spaced grid, e.g. for building p0_values/ple_values/sigma_values."""
        if n <= 1:
            return (min_val,)
        return tuple(np.linspace(min_val, max_val, n).tolist())
