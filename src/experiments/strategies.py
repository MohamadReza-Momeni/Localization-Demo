import numpy as np
from src.localization.scipy_solver import SciPyLocalizationSolver
from src.localization.ipopt_solver import IPOPTSolver
from src.localization.particle_filter_solver import ParticleFilterSolver
from src.experiments.context import RunContext


class SolverRegistry:
    """OCP: To add a new solver, just add a new method and register it in the dictionary."""

    def __init__(self, x_range, y_range):
        self.scipy_solver = SciPyLocalizationSolver()
        self.ipopt_solver = IPOPTSolver()
        self.pf_solver = ParticleFilterSolver(x_bounds=x_range, y_bounds=y_range)

        # THE STRATEGY DICTIONARY
        self.strategies = {
            "vanilla": self._run_vanilla,
            "weighted": self._run_weighted,
            "ipopt": self._run_ipopt,
            "weighted_ipopt": self._run_weighted_ipopt,
            "particle_filter": self._run_pf
        }

    def execute_solver(self, solver_name: str, context: RunContext):
        if solver_name not in self.strategies:
            raise ValueError(f"Unknown solver: {solver_name}")
        # Execute the specific strategy dynamically!
        return self.strategies[solver_name](context)

    # --- INDIVIDUAL STRATEGIES ---

    def _compute_weights(self, anchors, baseline_guess):
        # Weight anchors by proximity to the current position estimate, not to the
        # anchor centroid — closer anchors are generally more reliable in log-distance
        # RSSI models. BUT: keep this mild. Squaring or otherwise over-sharpening this
        # can let one nearby anchor dominate the objective, making the 2-anchor problem
        # nearly rank-deficient and letting the optimizer run off along the flat direction.
        d = np.linalg.norm(anchors - baseline_guess, axis=1)
        raw_weights = 1.0 / (d + 1.0)  # linear (not squared) inverse distance, +1 to soften near-zero d

        # Normalize so mean weight = 1: keeps the objective's overall scale stable across
        # runs and prevents any single anchor's weight from swamping the rest.
        raw_weights = raw_weights / np.mean(raw_weights)

        # Hard cap on the weight ratio so no anchor can be weighted more than 5x any other,
        # preserving enough multilateration geometry for the optimizer to stay well-conditioned.
        return np.clip(raw_weights, 0.2, 5.0)

    def _run_vanilla(self, ctx: RunContext):
        return self.scipy_solver.solve(ctx.anchors, ctx.distances, x_range=ctx.x_range, y_range=ctx.y_range)

    def _run_weighted(self, ctx: RunContext):
        weights = self._compute_weights(ctx.anchors, ctx.baseline_guess)
        return self.scipy_solver.solve(
            ctx.anchors, ctx.distances, weights=weights, x_range=ctx.x_range, y_range=ctx.y_range
        )

    def _run_ipopt(self, ctx: RunContext):
        sol = self.ipopt_solver.solve(
            ctx.anchors, ctx.distances, ref_power=ctx.p0, ple=ctx.ple,
            x0=ctx.baseline_guess, x_range=ctx.x_range, y_range=ctx.y_range
        )
        sol["success"] = sol["info"].get("status", -1) in [0, 1]
        return sol

    def _run_weighted_ipopt(self, ctx: RunContext):
        weights = self._compute_weights(ctx.anchors, ctx.baseline_guess)
        sol = self.ipopt_solver.solve(
            ctx.anchors, ctx.distances, ref_power=ctx.p0, ple=ctx.ple,
            weights=weights, x0=ctx.baseline_guess, x_range=ctx.x_range, y_range=ctx.y_range
        )
        sol["success"] = sol["info"].get("status", -1) in [0, 1]
        return sol

    def _run_pf(self, ctx: RunContext):
        return self.pf_solver.solve(ctx.anchors, ctx.distances, x0=ctx.baseline_guess)