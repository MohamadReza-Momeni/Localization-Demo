"""
src/experiments/clustering_config.py

E) Config for the anchor-clustering geometry study: "3 scenarios,
20m-40m clustering".

Structurally this is the geometry-only counterpart to sweep_config.py's
signal-only sweep: sweep_config.py holds anchor geometry fixed per
replication and varies P0/beta/sigma; ClusteringConfig holds P0/beta/sigma
fixed (by default a single representative value each — override with more
values in p0_values/ple_values/sigma_values if you also want to cross the
geometry study with a signal grid) and varies anchor CLUSTER SPACING
instead, so any error difference across scenarios is attributable to
geometry (GDOP), not to signal conditions changing underneath you.
"""
from dataclasses import dataclass, field
from src.localization.ipopt_params import IPOPTHyperparams


@dataclass(frozen=True)
class ClusteringScenario:
    name: str
    cluster_spacing: float  # meters; the intra-cluster anchor spacing knob
    n_clusters: int = 2


# The 3 scenarios from the professor's note: tight/medium/loose clustering
# spanning 20m-40m intra-cluster spacing. Adjust n_clusters per-scenario if
# you want e.g. more clusters at looser spacing.
DEFAULT_SCENARIOS = (
    ClusteringScenario(name="tight_20m", cluster_spacing=20.0, n_clusters=2),
    ClusteringScenario(name="medium_30m", cluster_spacing=30.0, n_clusters=2),
    ClusteringScenario(name="loose_40m", cluster_spacing=40.0, n_clusters=2),
)


@dataclass(frozen=True)
class ClusteringConfig:
    anchor_count: int = 6
    target_count: int = 1
    x_range: tuple[float, float] = (0, 1000)
    y_range: tuple[float, float] = (0, 1000)

    lat0: float = 35.7152
    lon0: float = 51.4043

    scenarios: tuple[ClusteringScenario, ...] = DEFAULT_SCENARIOS

    # Signal parameters held fixed (single-element tuples) by default so the
    # study isolates the geometry effect. Pass more values in each tuple if
    # you deliberately want to cross clustering with a signal grid too —
    # cost scales multiplicatively, same caveat as SweepConfig.total_solver_calls().
    p0_values: tuple[float, ...] = (-40.0,)
    ple_values: tuple[float, ...] = (3.0,)
    sigma_values: tuple[float, ...] = (2.0,)

    # Heterogeneous (distance-dependent) noise; het_factor=0.0 -> homogeneous
    # (unchanged). See src/signal/rssi.py and ExperimentConfig for details.
    het_factor: float = 0.0
    het_reference_distance: float = 100.0

    # Master RNG seed. None -> OS entropy per replication; int -> reproducible.
    base_seed: int | None = None

    # Independent anchor-cluster-layout + target draws per scenario — the
    # Monte Carlo repeat count needed for confidence bounds on the geometry
    # comparison (same role as SweepConfig.replications).
    replications: int = 30

    solvers: tuple[str, ...] = ("vanilla", "weighted", "ipopt", "weighted_ipopt", "particle_filter")
    ipopt_params: IPOPTHyperparams = field(default_factory=IPOPTHyperparams)
    ampl_options: dict = field(default_factory=dict)

    def grid_size(self) -> int:
        return len(self.p0_values) * len(self.ple_values) * len(self.sigma_values)

    def total_solver_calls(self) -> int:
        return (
            self.replications * self.target_count * len(self.scenarios)
            * self.grid_size() * len(self.solvers)
        )
