from dataclasses import dataclass, field
from src.localization.ipopt_params import IPOPTHyperparams


@dataclass(frozen=True)
class ExperimentConfig:
    """SRP: Holds the configuration parameters for a batch of simulations."""
    anchor_count: int = 6
    target_count: int = 1
    x_range: tuple[float, float] = (0, 1000)
    y_range: tuple[float, float] = (0, 1000)

    lat0: float = 35.7152
    lon0: float = 51.4043

    p0_range: tuple[float, float] = (-50.0, 50.0)
    ple_range: tuple[float, float] = (2.0, 8.0)
    noise_range: tuple[float, float] = (0.0, 10.0)

    # Heterogeneous (distance-dependent) noise knob. het_factor=0.0 keeps the
    # original homogeneous model (every anchor shares one sigma); het_factor>0
    # makes sigma grow with range, so the weighted solvers have a reliability
    # signal to exploit. See src/signal/rssi.py. Off by default so existing
    # studies/CSVs are unchanged unless explicitly enabled.
    het_factor: float = 0.0
    het_reference_distance: float = 100.0

    # Master RNG seed for reproducibility. None -> fresh OS entropy per run
    # (the historical non-reproducible behaviour); an int makes the whole study
    # deterministic (anchors, targets, and RSSI noise) — see run_support.py.
    base_seed: int | None = None

    solvers: tuple[str, ...] = ("vanilla", "weighted", "ipopt", "weighted_ipopt", "particle_filter")

    # IPOPT internals are now an explicit, loggable project parameter rather
    # than being hardcoded inside IPOPTSolver.solve. See ipopt_params.py.
    ipopt_params: IPOPTHyperparams = field(default_factory=IPOPTHyperparams)

    # Raw per-solver AMPL option strings for the ampl_bonmin/ampl_scip strategies.
    # e.g. {"scip": "limits/gap=0.01"}. See ampl_solver.py for why these are
    # passed through raw rather than typed.
    ampl_options: dict = field(default_factory=dict)
