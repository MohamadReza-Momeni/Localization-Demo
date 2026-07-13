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


class SimulationTask:

    def __init__(self, config: ExperimentConfig):
        self.config = config
        # Initialize our OCP Strategy Registry
        self.registry = SolverRegistry(config.x_range, config.y_range)

    def execute(self, run_id: int) -> list[dict]:
        p0, ple, sigma = self._sample_environment_variables()
        anchors, targets = self._generate_scenario(run_id)

        # Serialize the actual anchor positions used in this run. Anchors are generated
        # with a random seed (not derived from run_id), so they can no longer be
        # reproduced later just by re-seeding — we must persist the exact coordinates
        # so map_generator.py can plot the anchors that were really used.
        anchors_json = json.dumps(anchors.tolist())

        model = RSSIModel(reference_power=p0, path_loss_exponent=ple, noise_std=sigma)
        rssi_matrix = model.rssi_matrix(anchors, targets)

        results = []
        for target_id, true_position in enumerate(targets):
            distances = self._calculate_distances(rssi_matrix[:, target_id], p0, ple)

            # WARM START CALCULATION
            baseline_guess = self.registry.execute_solver("vanilla", RunContext(
                anchors, distances, None, p0, ple, self.config.x_range, self.config.y_range
            ))["solution"]

            # CREATE CONTEXT
            ctx = RunContext(
                anchors, distances, baseline_guess, p0, ple, self.config.x_range, self.config.y_range
            )

            # CLEAN EVALUATION LOOP
            for solver_name in self.config.solvers:
                # MAGIC HAPPENS HERE: No if statements!
                solution = self.registry.execute_solver(solver_name, ctx)

                est = solution["solution"]
                results.append({
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
                })

        return results

    # --- HELPER METHODS ---

    def _sample_environment_variables(self) -> tuple[float, float, float]:
        p0 = np.random.uniform(self.config.p0_range[0], self.config.p0_range[1])
        ple = np.random.uniform(self.config.ple_range[0], self.config.ple_range[1])
        sigma = np.random.uniform(self.config.noise_range[0], self.config.noise_range[1])
        return p0, ple, sigma

    def _generate_scenario(self, run_id: int):
        # Anchors and targets both use genuinely random seeds (seed=None -> OS entropy)
        # instead of fixed seeds derived from run_id. This means every run gets a
        # different anchor layout AND a different target position, even if you re-run
        # main.py/app.py with the same run_id. Safe under ProcessPoolExecutor/threads:
        # numpy's default_rng(None) pulls fresh OS entropy per process/call.
        anchors = AnchorGenerator(
            self.config.anchor_count, self.config.x_range, self.config.y_range, seed=None
        ).generate()

        targets = TargetGenerator(
            self.config.target_count, self.config.x_range, self.config.y_range, seed=None
        ).generate()

        return anchors, targets

    def _calculate_distances(self, rssi_values, p0, ple):
        return np.array([rssi_to_distance(rssi, p0, ple) for rssi in rssi_values])