import numpy as np
from src.localization.scipy_solver import SciPyLocalizationSolver
from src.localization.ipopt_solver import IPOPTSolver
from src.localization.ipopt_solver_new import IPOPTSolver as NewIPOPTSolver
from src.localization.ipopt_params import IPOPTHyperparams
from src.localization.particle_filter_solver import ParticleFilterSolver
from src.localization.ampl_solver import AMPLSolver
from src.experiments.context import RunContext


class SolverRegistry:
    """OCP: To add a new solver, just add a new method and register it in the dictionary."""

    def __init__(self, x_range, y_range):
        self.scipy_solver = SciPyLocalizationSolver()
        self.ipopt_solver = IPOPTSolver()
        self.ipopt_new_solver = NewIPOPTSolver()
        self.pf_solver = ParticleFilterSolver(x_bounds=x_range, y_bounds=y_range)
        self.ampl_solver = AMPLSolver(solver_name="bonmin")

        # THE STRATEGY DICTIONARY
        self.strategies = {
            "vanilla": self._run_vanilla,
            "weighted": self._run_weighted,
            "ipopt": self._run_ipopt,
            "weighted_ipopt": self._run_weighted_ipopt,
            "particle_filter": self._run_pf,
            "ampl_bonmin": self._run_ampl_bonmin,
            "ipopt_new": self._run_ipopt_new
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
        # NOTE: this is a DISTANCE-domain weighting — only used for the distance-domain
        # solver (weighted/scipy). Do not reuse this for weighted_ipopt (see
        # _compute_rssi_weights below) — that was a domain mismatch we corrected.
        d = np.linalg.norm(anchors - baseline_guess, axis=1)
        raw_weights = 1.0 / (d + 1.0)  # linear (not squared) inverse distance, +1 to soften near-zero d

        # Normalize so mean weight = 1: keeps the objective's overall scale stable across
        # runs and prevents any single anchor's weight from swamping the rest.
        raw_weights = raw_weights / np.mean(raw_weights)

        # Hard cap on the weight ratio so no anchor can be weighted more than 5x any other,
        # preserving enough multilateration geometry for the optimizer to stay well-conditioned.
        return np.clip(raw_weights, 0.2, 5.0)

    def _compute_rssi_weights(self, distances, p0, ple, noise_floor=-100.0):
        # RSSI-domain weighting for weighted_ipopt: derived from the MEASURED RSSI itself
        # (recovered from distances/p0/ple, the same inversion RSSIDomainProblem does
        # internally), not from distance to an uncertain baseline guess. This keeps the
        # weighting consistent with the domain the objective actually lives in.
        #
        # Physical justification: anchors reporting a stronger (less negative) RSSI are
        # generally further from the receiver's noise floor and thus more reliable —
        # this is a real hardware effect, independent of geometric distance to any
        # particular candidate position.
        #
        # CAVEAT: this simulator's RSSIModel currently applies the SAME noise_std to every
        # anchor regardless of signal strength or distance (see rssi.py). So even with a
        # theoretically-correct RSSI-domain weight, there is currently no real per-anchor
        # reliability difference in the data for this weighting to exploit — expect
        # weighted_ipopt to converge close to plain ipopt's behavior unless the noise
        # model is made heterogeneous (e.g. via distance_noise_growth).
        distances = np.asarray(distances, dtype=float)
        rssi_meas = p0 - 10.0 * ple * np.log10(distances + 1e-9)

        reliability = np.clip(rssi_meas - noise_floor, 1.0, None)
        raw_weights = reliability / np.mean(reliability)

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
        weights = self._compute_rssi_weights(ctx.distances, ctx.p0, ctx.ple)
        sol = self.ipopt_solver.solve(
            ctx.anchors, ctx.distances, ref_power=ctx.p0, ple=ctx.ple,
            weights=weights, x0=ctx.baseline_guess, x_range=ctx.x_range, y_range=ctx.y_range
        )
        sol["success"] = sol["info"].get("status", -1) in [0, 1]
        return sol

    def _run_pf(self, ctx: RunContext):
        return self.pf_solver.solve(ctx.anchors, ctx.distances, x0=ctx.baseline_guess)

    def _run_ampl_bonmin(self, ctx: RunContext):
        return self.ampl_solver.solve(ctx)

    def _run_ipopt_new(self, ctx: RunContext):
        # We pass exact Hessian hyperparameters to the new mathematically robust solver
        hyp = IPOPTHyperparams(hessian_approximation="exact") 
        sol = self.ipopt_new_solver.solve(
            ctx.anchors, ctx.distances, ref_power=ctx.p0, ple=ctx.ple,
            x0=ctx.baseline_guess, x_range=ctx.x_range, y_range=ctx.y_range,
            hyperparams=hyp
        )
        sol["success"] = sol.get("success", False)
        return sol