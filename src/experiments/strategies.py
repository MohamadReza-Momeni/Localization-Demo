import time
import numpy as np
from src.localization.scipy_solver import SciPyLocalizationSolver
from src.localization.ipopt_solver import IPOPTSolver
from src.localization.ampl_solver import AMPLSolver
from src.localization.particle_filter_solver import ParticleFilterSolver
from src.experiments.context import RunContext


class SolverRegistry:
    """OCP: To add a new solver, just add a new method and register it in the dictionary."""

    def __init__(self, x_range, y_range):
        self.scipy_solver = SciPyLocalizationSolver()
        self.ipopt_solver = IPOPTSolver()
        self.ampl_solver = AMPLSolver()
        self.pf_solver = ParticleFilterSolver(x_bounds=x_range, y_bounds=y_range)

        # THE STRATEGY DICTIONARY
        # NOTE: "ampl_cbc"/"ampl_cuopt" are intentionally NOT registered here —
        # both are LP/MIP-oriented solvers that cannot represent this problem's
        # nonlinear RSSI objective. See ampl_solver.py's SOLVER_CAPABILITIES.
        # They remain callable directly via AMPLSolver().solve(solver_name=...)
        # for anyone who wants to see the explicit error, e.g. in a demo script.
        #
        # "ampl_ipopt" (in addition to the native cyipopt-backed "ipopt") lets a
        # D)-style comparison isolate the SOLVER from the INTERFACE: "ipopt" vs
        # "ampl_ipopt" should converge to essentially the same objective given the
        # same starts (same underlying algorithm, different binding), whereas
        # "ipopt" vs "ampl_bonmin"/"ampl_scip" isolates genuine solver differences.
        self.strategies = {
            "vanilla": self._run_vanilla,
            "weighted": self._run_weighted,
            "ipopt": self._run_ipopt,
            "weighted_ipopt": self._run_weighted_ipopt,
            "ampl_ipopt": self._run_ampl_ipopt,
            "ampl_bonmin": self._run_ampl_bonmin,
            "ampl_scip": self._run_ampl_scip,
            "particle_filter": self._run_pf,
        }

    def execute_solver(self, solver_name: str, context: RunContext):
        if solver_name not in self.strategies:
            raise ValueError(f"Unknown solver: {solver_name}")

        # Wall-clock timing around the actual solve — deliberately measured here
        # (one place, uniformly, for every strategy) rather than inside each
        # solver, so every solver is timed the same way regardless of whether its
        # own .solve() internally does multi-start, AMPL round-trips, etc.
        # This is what feeds the "accuracy vs cost" side of the D) solver
        # comparison (see solver_comparison_report.py) — RMSE alone doesn't tell
        # you that SCIP's global search costs 50x what IPOPT's local search does.
        start = time.perf_counter()
        result = self.strategies[solver_name](context)
        result["solve_time_sec"] = time.perf_counter() - start
        return result

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
            x0=ctx.baseline_guess, x_range=ctx.x_range, y_range=ctx.y_range,
            hyperparams=ctx.ipopt_params,
        )
        sol["success"] = sol["info"].get("status", -1) in [0, 1]
        return sol

    def _run_weighted_ipopt(self, ctx: RunContext):
        weights = self._compute_rssi_weights(ctx.distances, ctx.p0, ctx.ple)
        sol = self.ipopt_solver.solve(
            ctx.anchors, ctx.distances, ref_power=ctx.p0, ple=ctx.ple,
            weights=weights, x0=ctx.baseline_guess, x_range=ctx.x_range, y_range=ctx.y_range,
            hyperparams=ctx.ipopt_params,
        )
        sol["success"] = sol["info"].get("status", -1) in [0, 1]
        return sol

    def _run_ampl_ipopt(self, ctx: RunContext):
        # Same RSSI-domain NLP, same starting-point strategy, routed through AMPL
        # instead of cyipopt directly. Comparing this against "ipopt" isolates
        # interface/binding effects (AMPL's presolve, scaling, NL-file round trip)
        # from genuine algorithmic differences — the latter is what "ipopt" vs
        # "ampl_bonmin"/"ampl_scip" actually measures.
        sol = self.ampl_solver.solve(
            ctx.anchors, ctx.distances, ref_power=ctx.p0, ple=ctx.ple,
            x0=ctx.baseline_guess, x_range=ctx.x_range, y_range=ctx.y_range,
            solver_name="ipopt", solver_options=ctx.ampl_options.get("ipopt", ""),
        )
        return sol

    def _run_ampl_bonmin(self, ctx: RunContext):
        sol = self.ampl_solver.solve(
            ctx.anchors, ctx.distances, ref_power=ctx.p0, ple=ctx.ple,
            x0=ctx.baseline_guess, x_range=ctx.x_range, y_range=ctx.y_range,
            solver_name="bonmin", solver_options=ctx.ampl_options.get("bonmin", ""),
        )
        return sol

    def _run_ampl_scip(self, ctx: RunContext):
        sol = self.ampl_solver.solve(
            ctx.anchors, ctx.distances, ref_power=ctx.p0, ple=ctx.ple,
            x0=ctx.baseline_guess, x_range=ctx.x_range, y_range=ctx.y_range,
            solver_name="scip", solver_options=ctx.ampl_options.get("scip", ""),
        )
        return sol

    def _run_pf(self, ctx: RunContext):
        return self.pf_solver.solve(ctx.anchors, ctx.distances, x0=ctx.baseline_guess)
