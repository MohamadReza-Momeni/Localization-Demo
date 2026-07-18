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
        self.registry = SolverRegistry(config.x_range, config.y_range)

    def execute(self, params: tuple[int, float, float, float]) -> list[dict]:
        # NEW: Unpack the exact parameters dictated by the Grid Search executor
        run_id, p0, ple, sigma = params
        
        # Generate reproducible map based on run_id.
        anchors, targets = self._generate_scenario(run_id)

        anchors_json = json.dumps(anchors.tolist())

        model = RSSIModel(reference_power=p0, path_loss_exponent=ple, noise_std=sigma)
        rssi_matrix = model.rssi_matrix(anchors, targets, samples=self.config.samples_per_anchor)

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
                    "samples_per_anchor": self.config.samples_per_anchor,
                    "map_width": self.config.x_range[1],
                    "map_height": self.config.y_range[1],
                    "lat0": self.config.lat0,
                    "lon0": self.config.lon0,
                    "true_x": true_position[0], "true_y": true_position[1],
                    "est_x": est[0], "est_y": est[1], "error": euclidean_error(true_position, est),
                    "success": solution["success"], 
                    "p0": p0, "ple": ple, "sigma": sigma, # Exact grid parameters used
                    "anchors": anchors_json,
                })

        return results

    # --- HELPER METHODS ---

    def _generate_scenario(self, run_id: int):
        # Grid Search requires that we test the *exact same* room layout across different noise levels!
        # Therefore, we link the generation seed strictly to the run_id.
        anchors = AnchorGenerator(
            self.config.anchor_count, self.config.x_range, self.config.y_range, seed=42 + run_id
        ).generate()

        targets = TargetGenerator(
            self.config.target_count, self.config.x_range, self.config.y_range, seed=1 + run_id
        ).generate()

        return anchors, targets

    def _calculate_distances(self, rssi_values, p0, ple):
        return np.array([rssi_to_distance(rssi, p0, ple) for rssi in rssi_values])