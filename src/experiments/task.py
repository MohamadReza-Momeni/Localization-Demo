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

            # --- CRLB CALCULATION ---
            # Calculate the theoretical lowest possible error for this specific target
            crlb_error = self._calculate_crlb(
                anchors, true_position, ple, sigma, self.config.samples_per_anchor
            )

            # WARM START CALCULATION
            baseline_guess = self.registry.execute_solver("vanilla", RunContext(
                anchors, distances, None, p0, ple, self.config.x_range, self.config.y_range
            ))["solution"]

            ctx = RunContext(
                anchors, distances, baseline_guess, p0, ple, self.config.x_range, self.config.y_range
            )

            # 1. SOLVER LOOP (Builds the original table)
            for solver_name in self.config.solvers:
                solution = self.registry.execute_solver(solver_name, ctx)
                est = solution["solution"]
                solver_results.append({
                    "run_id": run_id, "target_id": target_id, "solver": solver_name,
                    "anchor_count": self.config.anchor_count,
                    "samples_per_anchor": self.config.samples_per_anchor,
                    "map_width": self.config.x_range[1], "map_height": self.config.y_range[1],
                    "lat0": self.config.lat0, "lon0": self.config.lon0,
                    "true_x": true_position[0], "true_y": true_position[1],
                    "est_x": est[0], "est_y": est[1], 
                    "error": euclidean_error(true_position, est),
                    "crlb": crlb_error, # NEW: The theoretical limit
                    "success": solution["success"], 
                    "p0": p0, "ple": ple, "sigma": sigma,
                    "anchors": anchors_json,
                })

            # 2. RAW MEASUREMENT LOOP (Builds the new table)
            for anchor_id, anchor in enumerate(anchors):
                true_distance = np.linalg.norm(anchor - true_position)
                est_distance = distances[anchor_id] 
                
                for sample_id, raw_rssi in enumerate(raw_samples[anchor_id, target_id]):
                    measurement_results.append({
                        "run_id": run_id,
                        "target_id": target_id,
                        "anchor_id": anchor_id,
                        "sample_id": sample_id,
                        "map_width": self.config.x_range[1],
                        "map_height": self.config.y_range[1],
                        "lat0": self.config.lat0,
                        "lon0": self.config.lon0,
                        "true_x": true_position[0],
                        "true_y": true_position[1],
                        "anchor_x": anchor[0],
                        "anchor_y": anchor[1],
                        "p0": p0,
                        "ple": ple,
                        "sigma": sigma,
                        "raw_rssi": raw_rssi,
                        "true_distance": true_distance,
                        "est_distance": est_distance,
                        "error_distance": abs(true_distance - est_distance)
                    })

        return solver_results, measurement_results

    # --- HELPER METHODS ---

    def _generate_scenario(self, run_id: int):
        anchors = AnchorGenerator(
            self.config.anchor_count, self.config.x_range, self.config.y_range, seed=42 + run_id
        ).generate()

        targets = TargetGenerator(
            self.config.target_count, self.config.x_range, self.config.y_range, seed=1 + run_id
        ).generate()

        return anchors, targets

    def _calculate_distances(self, rssi_values, p0, ple):
        return np.array([rssi_to_distance(rssi, p0, ple) for rssi in rssi_values])
        
    def _calculate_crlb(self, anchors, target_pos, ple, sigma, samples):
        """Calculates the Cramer-Rao Lower Bound (CRLB) in meters."""
        if sigma <= 0:
            return 0.0 # No noise = perfect accuracy limit
            
        # Incorporate the Central Limit Theorem:
        # Taking multiple measurements reduces the effective noise variance
        effective_sigma = sigma / np.sqrt(samples)
        
        # Log-distance path loss derivative constant
        K = (10 * ple) / (effective_sigma * np.log(10))
        
        FIM = np.zeros((2, 2))
        x, y = target_pos
        
        for ax, ay in anchors:
            dx = x - ax
            dy = y - ay
            d_sq = dx**2 + dy**2
            
            # Prevent division by zero if target is exactly on an anchor
            d_sq = max(d_sq, 1e-12) 
            
            coeff = (K / d_sq)**2
            
            FIM[0, 0] += coeff * (dx**2)
            FIM[1, 1] += coeff * (dy**2)
            FIM[0, 1] += coeff * (dx * dy)
            FIM[1, 0] += coeff * (dx * dy)
            
        try:
            # Invert the Fisher Information Matrix
            crlb_matrix = np.linalg.inv(FIM)
            # RMSE Lower Bound = sqrt(Trace of CRLB matrix)
            return np.sqrt(max(np.trace(crlb_matrix), 0.0))
        except np.linalg.LinAlgError:
            # If the FIM is singular (e.g., all anchors in a perfectly straight line)
            return float('inf')