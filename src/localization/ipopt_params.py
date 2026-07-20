from dataclasses import dataclass

@dataclass(frozen=True)
class IPOPTHyperparams:
    tol: float = 1e-6                  
    acceptable_tol: float = 1e-4       
    acceptable_iter: int = 15          
    max_iter: int = 500
    hessian_approximation: str = "limited-memory"  # or "exact"
    mu_strategy: str = "monotone"                  # or "adaptive"
    print_level: int = 0
    starting_points: tuple[str, ...] = ("warm_start", "anchor_centroid", "random_point")
    multi_start: bool = True  
    fixed_initial_point: tuple[float, float] | None = None

    def __post_init__(self):
        if self.acceptable_tol < self.tol:
            raise ValueError(f"acceptable_tol ({self.acceptable_tol}) must be >= tol ({self.tol})")
        valid_points = {"warm_start", "anchor_centroid", "random_point"}
        unknown = set(self.starting_points) - valid_points
        if unknown:
            raise ValueError(f"Unknown starting point strategy label(s): {unknown}")
        if len(self.starting_points) == 0:
            raise ValueError("starting_points cannot be empty")

    def as_ipopt_options(self) -> dict:
        return {
            "sb": "yes",
            "print_level": self.print_level,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "acceptable_tol": self.acceptable_tol,
            "acceptable_iter": self.acceptable_iter,
            "hessian_approximation": self.hessian_approximation,
            "mu_strategy": self.mu_strategy,
        }