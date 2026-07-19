import json
import numpy as np
from src.evaluation.metrics import euclidean_error
from src.scenario.anchors import AnchorGenerator
from src.scenario.targets import TargetGenerator
from src.signal.distance import rssi_to_distance
from src.signal.rssi import RSSIModel

from src.experiments.config import ExperimentConfig
from src.experiments.context import RunContext
from src.experiments.strategies import SolverRegistry
from src.experiments.run_support import derive_seeds, augment_ipopt_ampl_columns


class SimulationTask:

    def __init__(self, config: ExperimentConfig):
        self.config = config
        # Initialize our OCP Strategy Registry
        self.registry = SolverRegistry(config.x_range, config.y_range)

    def execute(self, run_id: int) -> list[dict]:
        # Four independent seed streams per run: environment sampling, anchors,
        # targets, and RSSI noise. With config.base_seed set this makes run_id's
        # entire realisation reproducible; with base_seed=None each stream draws
        # fresh OS entropy (the historical behaviour). See run_support.py.
        env_seed, anchor_seed, target_seed, noise_seed = derive_seeds(
            self.config.base_seed, run_id, 4
        )

        p0, ple, sigma = self._sample_environment_variables(env_seed)
        anchors, targets = self._generate_scenario(anchor_seed, target_seed)

        # Serialize the actual anchor positions used in this run so map_generator.py
        # can plot the anchors that were really used (needed either way, but now the
        # layout is also reproducible from run_id when base_seed is set).
        anchors_json = json.dumps(anchors.tolist())

        model = RSSIModel(
            reference_power=p0, path_loss_exponent=ple, noise_std=sigma,
            het_factor=self.config.het_factor,
            het_reference_distance=self.config.het_reference_distance,
            rng=np.random.default_rng(noise_seed),
        )
        rssi_matrix = model.rssi_matrix(anchors, targets)

        results = []
        for target_id, true_position in enumerate(targets):
            distances = self._calculate_distances(rssi_matrix[:, target_id], p0, ple)

            # WARM START CALCULATION
            baseline_guess = self.registry.execute_solver("vanilla", RunContext(
                anchors, distances, None, p0, ple, self.config.x_range, self.config.y_range,
                ipopt_params=self.config.ipopt_params, ampl_options=self.config.ampl_options,
            ))["solution"]

            # CREATE CONTEXT
            ctx = RunContext(
                anchors, distances, baseline_guess, p0, ple, self.config.x_range, self.config.y_range,
                ipopt_params=self.config.ipopt_params, ampl_options=self.config.ampl_options,
            )

            # CLEAN EVALUATION LOOP
            for solver_name in self.config.solvers:
                # MAGIC HAPPENS HERE: No if statements!
                solution = self.registry.execute_solver(solver_name, ctx)

                est = solution["solution"]
                row = {
                    "run_id": run_id, "target_id": target_id, "solver": solver_name,
                    "anchor_count": self.config.anchor_count,
                    "map_width": self.config.x_range[1],
                    "map_height": self.config.y_range[1],
                    "lat0": self.config.lat0,
                    "lon0": self.config.lon0,
                    "true_x": true_position[0], "true_y": true_position[1],
                    "est_x": est[0], "est_y": est[1], "error": euclidean_error(true_position, est),
                    "success": solution["success"], "p0": p0, "ple": ple, "sigma": sigma,
                    "anchors": anchors_json,
                    # D) solver-comparison: wall-clock cost per solver, added by
                    # SolverRegistry.execute_solver's uniform timing wrapper.
                    "solve_time_sec": solution["solve_time_sec"],
                    # RSSI-domain objective at the returned solution (None for the
                    # SciPy solvers). Logged for consistency with the sweep/clustering
                    # tasks so failed-run audits are possible after the fact.
                    "objective": solution.get("objective"),
                }

                # IPOPT-family / AMPL-family bookkeeping columns (which start won,
                # internal hyperparameters, underlying AMPL solver) — see
                # run_support.augment_ipopt_ampl_columns.
                augment_ipopt_ampl_columns(row, solution, self.config.ipopt_params)

                results.append(row)

        return results

    # --- HELPER METHODS ---

    def _sample_environment_variables(self, seed) -> tuple[float, float, float]:
        # Drawn from a per-run seeded stream (reproducible when base_seed is set)
        # rather than the global np.random state.
        rng = np.random.default_rng(seed)
        p0 = rng.uniform(self.config.p0_range[0], self.config.p0_range[1])
        ple = rng.uniform(self.config.ple_range[0], self.config.ple_range[1])
        sigma = rng.uniform(self.config.noise_range[0], self.config.noise_range[1])
        return p0, ple, sigma

    def _generate_scenario(self, anchor_seed, target_seed):
        # Anchors and targets are seeded from the per-run streams derived in
        # execute() rather than seed=None. With config.base_seed set this makes
        # the layout + target reproducible from run_id; with base_seed=None each
        # SeedSequence is spawned from fresh OS entropy, preserving the old
        # every-run-different behaviour. Safe under ProcessPoolExecutor: each
        # child seed is independent and self-contained (no shared global state).
        anchors = AnchorGenerator(
            self.config.anchor_count, self.config.x_range, self.config.y_range, seed=anchor_seed
        ).generate()

        targets = TargetGenerator(
            self.config.target_count, self.config.x_range, self.config.y_range, seed=target_seed
        ).generate()

        return anchors, targets

    def _calculate_distances(self, rssi_values, p0, ple):
        return np.array([rssi_to_distance(rssi, p0, ple) for rssi in rssi_values])
