import json
import numpy as np
from src.evaluation.metrics import euclidean_error
from src.evaluation.crlb import calculate_crlb
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

    def execute(self, params: tuple[int, float, float, float]) -> tuple[list[dict], list[dict]]:
        run_id, p0, ple, sigma = params
        
        anchors, targets = self._generate_scenario(run_id)
        anchors_json = json.dumps(anchors.tolist())

        model = RSSIModel(reference_power=p0, path_loss_exponent=ple, noise_std=sigma)
        rssi_matrix, raw_samples = model.rssi_matrix(anchors, targets, samples=self.config.samples_per_anchor)

        solver_results = []
        measurement_results = []

        for target_id, true_position in enumerate(targets):
            distances = self._calculate_distances(rssi_matrix[:, target_id], p0, ple)
            
            crlb_error = calculate_crlb(anchors, true_position, ple, sigma, self.config.samples_per_anchor)

            baseline_guess = self.registry.execute_solver("vanilla", RunContext(
                anchors, distances, None, p0, ple, self.config.x_range, self.config.y_range
            ))["solution"]

            ctx = RunContext(
                anchors, distances, baseline_guess, p0, ple, self.config.x_range, self.config.y_range
            )

            # 1. SOLVER LOOP
            for solver_name in self.config.solvers:
                solution = self.registry.execute_solver(solver_name, ctx)
                record = self._build_solver_record(
                    run_id, target_id, solver_name, true_position, solution, 
                    crlb_error, p0, ple, sigma, anchors_json
                )
                solver_results.append(record)

            # 2. RAW MEASUREMENT LOOP
            measurements = self._build_measurement_records(
                run_id, target_id, anchors, true_position, distances, raw_samples[:, target_id], p0, ple, sigma
            )
            measurement_results.extend(measurements)

        return solver_results, measurement_results

    # --- HELPER METHODS ---

    def _generate_scenario(self, run_id: int):
        anchors = AnchorGenerator(self.config.anchor_count, self.config.x_range, self.config.y_range, seed=42 + run_id).generate()
        targets = TargetGenerator(self.config.target_count, self.config.x_range, self.config.y_range, seed=1 + run_id).generate()
        return anchors, targets

    def _calculate_distances(self, rssi_values, p0, ple):
        return np.array([rssi_to_distance(rssi, p0, ple) for rssi in rssi_values])

    def _build_solver_record(self, run_id, target_id, solver_name, true_pos, solution, crlb, p0, ple, sigma, anchors_json):
        est = solution["solution"]
        return {
            "run_id": run_id, "target_id": target_id, "solver": solver_name,
            "anchor_count": self.config.anchor_count,
            "samples_per_anchor": self.config.samples_per_anchor,
            "map_width": self.config.x_range[1], "map_height": self.config.y_range[1],
            "lat0": self.config.lat0, "lon0": self.config.lon0,
            "true_x": true_pos[0], "true_y": true_pos[1],
            "est_x": est[0], "est_y": est[1], 
            "error": euclidean_error(true_pos, est),
            "crlb": crlb, 
            "success": solution["success"], 
            "p0": p0, "ple": ple, "sigma": sigma,
            "anchors": anchors_json,
        }

    def _build_measurement_records(self, run_id, target_id, anchors, true_pos, distances, raw_samples, p0, ple, sigma):
        records = []
        for anchor_id, anchor in enumerate(anchors):
            true_dist = np.linalg.norm(anchor - true_pos)
            est_dist = distances[anchor_id] 
            
            for sample_id, raw_rssi in enumerate(raw_samples[anchor_id]):
                records.append({
                    "run_id": run_id, "target_id": target_id, "anchor_id": anchor_id, "sample_id": sample_id,
                    "map_width": self.config.x_range[1], "map_height": self.config.y_range[1],
                    "lat0": self.config.lat0, "lon0": self.config.lon0,
                    "true_x": true_pos[0], "true_y": true_pos[1],
                    "anchor_x": anchor[0], "anchor_y": anchor[1],
                    "p0": p0, "ple": ple, "sigma": sigma,
                    "raw_rssi": raw_rssi,
                    "true_distance": true_dist, "est_distance": est_dist,
                    "error_distance": abs(true_dist - est_dist)
                })
        return records