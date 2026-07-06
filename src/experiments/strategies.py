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

    def _compute_weights(self, anchors):
        d = np.linalg.norm(anchors - np.mean(anchors, axis=0), axis=1)
        return 1.0 / (d + 1e-6)

    def _run_vanilla(self, ctx: RunContext):
        return self.scipy_solver.solve(ctx.anchors, ctx.distances)

    def _run_weighted(self, ctx: RunContext):
        weights = self._compute_weights(ctx.anchors)
        return self.scipy_solver.solve(ctx.anchors, ctx.distances, weights=weights)

    def _run_ipopt(self, ctx: RunContext):
        sol = self.ipopt_solver.solve(
            ctx.anchors, ctx.distances, ref_power=ctx.p0, ple=ctx.ple, 
            x0=ctx.baseline_guess, x_range=ctx.x_range, y_range=ctx.y_range
        )
        sol["success"] = sol["info"].get("status", -1) in [0, 1]
        return sol

    def _run_weighted_ipopt(self, ctx: RunContext):
        weights = self._compute_weights(ctx.anchors)
        sol = self.ipopt_solver.solve(
            ctx.anchors, ctx.distances, ref_power=ctx.p0, ple=ctx.ple, 
            weights=weights, x0=ctx.baseline_guess, x_range=ctx.x_range, y_range=ctx.y_range
        )
        sol["success"] = sol["info"].get("status", -1) in [0, 1]
        return sol

    def _run_pf(self, ctx: RunContext):
        return self.pf_solver.solve(ctx.anchors, ctx.distances, x0=ctx.baseline_guess)