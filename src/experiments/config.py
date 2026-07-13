from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """SRP: Holds the configuration parameters for a batch of simulations."""
    anchor_count: int = 6
    target_count: int = 1
    x_range: tuple[float, float] = (0, 1000)
    y_range: tuple[float, float] = (0, 1000)

    p0_range: tuple[float, float] = (-50.0, -30.0)
    ple_range: tuple[float, float] = (2.0, 4.0)
    noise_range: tuple[float, float] = (1.0, 3.0)

    solvers: tuple[str, ...] = ("vanilla", "weighted", "ipopt", "weighted_ipopt", "particle_filter")