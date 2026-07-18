from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    """SRP: Holds the configuration parameters for a batch parameter-sweep simulation."""
    anchor_count: int = 6
    samples_per_anchor: int = 1  # The new measurement averaging parameter!
    target_count: int = 1
    x_range: tuple[float, float] = (0, 1000)
    y_range: tuple[float, float] = (0, 1000)

    lat0: float = 35.7152
    lon0: float = 51.4043
    
    # RESTORED: Explicit value lists for parameter permutation sweeps
    p0_values: tuple[float, ...] = (-40.0,)
    ple_values: tuple[float, ...] = (2.2,)
    noise_values: tuple[float, ...] = (2.0,)
    
    solvers: tuple[str, ...] = ("vanilla", "weighted", "ipopt", "weighted_ipopt", "particle_filter")