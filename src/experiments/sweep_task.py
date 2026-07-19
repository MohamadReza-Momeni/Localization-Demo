"""
src/experiments/sweep_task.py

Grid-sweep counterpart to task.py's SimulationTask. Where SimulationTask
draws one random (P0, beta, sigma) per run, SweepTask holds one anchor
layout + target fixed and walks the FULL P0 x beta x sigma grid across it
— matching the "generate data -> same target -> next P0 -> next B" loop
from the professor's notes.

Reuses SolverRegistry/RunContext/RSSIModel exactly as task.py does, so the
solver behavior (including the IPOPT hyperparams from the previous
refactor) is identical between random-sampling and grid-sweep runs; only
how P0/beta/sigma are chosen differs.

Drop-in compatible with BatchExecutor: BatchExecutor(SweepTask(cfg)).run(
    run_count=cfg.replications) works exactly like it does for SimulationTask,
since SweepTask.execute(replication_id) has the same signature/return shape.
"""
import json
import numpy as np

from src.evaluation.metrics import euclidean_error
from src.scenario.anchors import AnchorGenerator
from src.scenario.targets import TargetGenerator
from src.signal.distance import rssi_to_distance
from src.signal.rssi import RSSIModel

from src.experiments.sweep_config import SweepConfig
from src.experiments.context import RunContext
from src.experiments.strategies import SolverRegistry
from src.experiments.run_support import derive_seeds, augment_ipopt_ampl_columns
from src.evaluation.crlb import crlb_rmse


class SweepTask:

    def __init__(self, config: SweepConfig):
        self.config = config
        self.registry = SolverRegistry(config.x_range, config.y_range)

    def execute(self, replication_id: int) -> list[dict]:
        """
        One replication = one fresh anchor layout + one set of targets, held
        fixed while every (P0, beta, sigma) grid combination is evaluated
        against them. This isolates signal-parameter effects from geometry
        effects, which random per-run sampling in task.py cannot do.
        """
        # Anchor/target seeds are per-replication; noise gets its own stream so
        # each grid point's noise realisation is reproducible yet independent.
        anchor_seed, target_seed, noise_seed = derive_seeds(
            self.config.base_seed, replication_id, 3
        )
        anchors, targets = self._generate_scenario(anchor_seed, target_seed)
        anchors_json = json.dumps(anchors.tolist())
        noise_rng = np.random.default_rng(noise_seed)

        results = []
        for target_id, true_position in enumerate(targets):
            for p0 in self.config.p0_values:
                for ple in self.config.ple_values:
                    for sigma in self.config.sigma_values:
                        results.extend(
                            self._evaluate_grid_point(
                                replication_id, target_id, true_position,
                                anchors, anchors_json, p0, ple, sigma, noise_rng,
                            )
                        )

        return results

    # --- HELPERS ---

    def _evaluate_grid_point(self, replication_id, target_id, true_position,
                              anchors, anchors_json, p0, ple, sigma, noise_rng):
        model = RSSIModel(
            reference_power=p0, path_loss_exponent=ple, noise_std=sigma,
            het_factor=self.config.het_factor,
            het_reference_distance=self.config.het_reference_distance,
            rng=noise_rng,
        )
        anchor_dists = np.array([np.linalg.norm(a - true_position) for a in anchors])
        rssi_values = np.array([model.rssi(d) for d in anchor_dists])
        distances = np.array([rssi_to_distance(r, p0, ple) for r in rssi_values])

        baseline_guess = self.registry.execute_solver("vanilla", RunContext(
            anchors, distances, None, p0, ple, self.config.x_range, self.config.y_range,
            ipopt_params=self.config.ipopt_params, ampl_options=self.config.ampl_options,
        ))["solution"]

        ctx = RunContext(
            anchors, distances, baseline_guess, p0, ple, self.config.x_range, self.config.y_range,
            ipopt_params=self.config.ipopt_params, ampl_options=self.config.ampl_options,
        )

        # Theoretical CRLB for this grid point (anchor geometry, true position,
        # beta, sigma). Independent of P0 by construction — see crlb.py's
        # module docstring. Under heterogeneous noise (het_factor>0) each anchor
        # has its own sigma, so pass the per-anchor sigma array to keep the CRLB
        # consistent with the noise actually generated; with het_factor=0 this
        # is a constant array == scalar sigma (identical to before).
        sigma_per_anchor = model.noise_std_at(anchor_dists)
        crlb_bound = crlb_rmse(anchors, true_position, ple, sigma_per_anchor)

        rows = []
        for solver_name in self.config.solvers:
            solution = self.registry.execute_solver(solver_name, ctx)
            est = solution["solution"]

            row = {
                "run_id": replication_id, "target_id": target_id, "solver": solver_name,
                "anchor_count": self.config.anchor_count,
                "map_width": self.config.x_range[1],
                "map_height": self.config.y_range[1],
                "lat0": self.config.lat0,
                "lon0": self.config.lon0,
                "true_x": true_position[0], "true_y": true_position[1],
                "est_x": est[0], "est_y": est[1], "error": euclidean_error(true_position, est),
                "success": solution["success"],
                # Explicit grid coordinates — NOT randomly sampled, unlike task.py.
                "p0": p0, "ple": ple, "sigma": sigma,
                "anchors": anchors_json,
                "crlb_rmse": crlb_bound,
                # D) solver-comparison: wall-clock cost per solver.
                "solve_time_sec": solution["solve_time_sec"],
                # RSSI-domain objective at the returned solution (None for the
                # SciPy solvers, which optimise a different distance-domain
                # objective). Logged so native-vs-AMPL comparisons can verify
                # which solver actually found the lower optimum, and to detect
                # boundary-stuck / worse-optimum outliers after the fact.
                "objective": solution.get("objective"),
            }

            augment_ipopt_ampl_columns(row, solution, self.config.ipopt_params)

            rows.append(row)

        return rows

    def _generate_scenario(self, anchor_seed, target_seed):
        anchors = AnchorGenerator(
            self.config.anchor_count, self.config.x_range, self.config.y_range, seed=anchor_seed
        ).generate()

        targets = TargetGenerator(
            self.config.target_count, self.config.x_range, self.config.y_range, seed=target_seed
        ).generate()

        return anchors, targets
