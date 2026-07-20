"""
src/localization/ipopt_params.py

Promotes IPOPT's internal solver mechanics (previously hardcoded inside
IPOPTSolver.solve) to a first-class, explicit project parameter — same
tier as P0 / beta (PLE) / sigma / the starting point.

Rationale (per professor's note): tuning epsilon/tol against acceptable_tol,
choosing which starting points feed multi-start, and swapping the Hessian
approximation strategy are all "internal runs of IPOPT" decisions that
should be visible and sweepable from the experiment layer, not buried in
solver internals where they can't be logged, swept, or reported in a paper.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class IPOPTHyperparams:
    # --- Convergence tolerances ---
    tol: float = 1e-6                  # main convergence tolerance ("epsilon")
    acceptable_tol: float = 1e-4       # looser fallback tolerance IPOPT accepts if
                                        # it stalls before reaching `tol`
    acceptable_iter: int = 15          # iterations at acceptable_tol before stopping
    max_iter: int = 500

    # --- Solver internals ---
    hessian_approximation: str = "limited-memory"  # or "exact"
    mu_strategy: str = "monotone"                  # or "adaptive"
    print_level: int = 0

    # --- Starting point strategy ---
    # Which candidate starts to run (subset/order of these three); the
    # solution with the lowest objective across all requested starts wins.
    starting_points: tuple[str, ...] = ("warm_start", "anchor_centroid", "random_point")
    multi_start: bool = True  # if False, only the first entry of starting_points is used

    # Optional explicit override for the "warm_start" candidate — e.g. to
    # reproduce a specific initial point used in a parallel MATLAB run.
    # When set, this replaces whatever x0 the caller passed in.
    fixed_initial_point: tuple[float, float] | None = None

    def __post_init__(self):
        if self.acceptable_tol < self.tol:
            # "epsilon کاملا بیشتر نباشه" — the acceptable (fallback) tolerance
            # is meant to be a *looser* safety net than the main tol, never a
            # stricter one. If acceptable_tol < tol, IPOPT's acceptable-point
            # fallback becomes meaningless (it would trigger later, not earlier,
            # than strict convergence) and likely indicates a swapped value.
            raise ValueError(
                f"acceptable_tol ({self.acceptable_tol}) must be >= tol ({self.tol}); "
                "acceptable_tol is meant to be a looser fallback threshold."
            )
        valid_points = {"warm_start", "anchor_centroid", "random_point"}
        unknown = set(self.starting_points) - valid_points
        if unknown:
            raise ValueError(f"Unknown starting point strategy label(s): {unknown}")
        if len(self.starting_points) == 0:
            raise ValueError("starting_points cannot be empty")

    def as_ipopt_options(self) -> dict:
        """IPOPT option dict for cyipopt.Problem.add_option(...) calls."""
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
