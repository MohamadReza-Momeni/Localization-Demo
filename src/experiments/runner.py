from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.evaluation.metrics import euclidean_error
from src.localization.scipy_solver import SciPyLocalizationSolver
from src.scenario.anchors import AnchorGenerator
from src.scenario.targets import TargetGenerator
from src.signal.distance import rssi_to_distance
from src.signal.rssi import RSSIModel
from src.localization.ipopt_solver import IPOPTSolver


@dataclass(frozen=True)
class ExperimentConfig:
    anchor_count: int = 6
    target_count: int = 1
    x_range: tuple[float, float] = (0, 1000)
    y_range: tuple[float, float] = (0, 1000)

    # UPDATED: Accept ranges instead of fixed values
    p0_range: tuple[float, float] = (-50.0, 50.0)
    ple_range: tuple[float, float] = (2.0, 8.0)
    noise_range: tuple[float, float] = (0.0, 10.0)

    solvers: tuple[str, ...] = ("vanilla", "weighted", "ipopt", "weighted_ipopt")


class ExperimentRunner:
    def __init__(
        self,
        anchor_count=6,
        target_count=1,
        x_range=(0, 1000),
        y_range=(0, 1000),
        p0_range=(-50.0, 50.0),
        ple_range=(2.0, 8.0),
        noise_range=(0.0, 10.0),
    ):
        self.config = ExperimentConfig(
            anchor_count=anchor_count,
            target_count=target_count,
            x_range=x_range,
            y_range=y_range,
            p0_range=p0_range,
            ple_range=ple_range,
            noise_range=noise_range,
        )
        
        # Instantiate solvers once
        self.scipy_solver = SciPyLocalizationSolver()
        self.ipopt_solver = IPOPTSolver()

    def run_batch(self, run_count):
        rows = []

        for run_id in range(run_count):
            # 1. RANDOMIZE VARIABLES FOR THIS RUN
            p0 = np.random.uniform(self.config.p0_range[0], self.config.p0_range[1])
            ple = np.random.uniform(self.config.ple_range[0], self.config.ple_range[1])
            sigma = np.random.uniform(self.config.noise_range[0], self.config.noise_range[1])
            
            # 2. CREATE MODEL WITH RANDOMIZED VARS
            model = RSSIModel(
                reference_power=p0, 
                path_loss_exponent=ple, 
                noise_std=sigma
            )

            # Generate Anchors and Targets
            anchor_seed = 42 + run_id
            target_seed = 1 + run_id
            anchors = AnchorGenerator(self.config.anchor_count, self.config.x_range, self.config.y_range, seed=anchor_seed).generate()
            targets = TargetGenerator(self.config.target_count, self.config.x_range, self.config.y_range, seed=target_seed).generate()

            # Get RSSI Matrix from the randomized model
            rssi_matrix = model.rssi_matrix(anchors, targets)

            for target_id, true_position in enumerate(targets):
                rssi_values = rssi_matrix[:, target_id]
                
                # Estimate distances using the randomized P0 and PLE
                distances = np.array([
                    rssi_to_distance(rssi, p0, ple) 
                    for rssi in rssi_values
                ])

                for solver_name in self.config.solvers:
                    if solver_name == "vanilla":
                        solution = self.scipy_solver.solve(anchors, distances)
                        success = solution["success"]
                    elif solver_name == "weighted":
                        weights = self._compute_weights(anchors)
                        solution = self.scipy_solver.solve(anchors, distances, weights=weights)
                        success = solution["success"]
                    elif solver_name == "ipopt":
                        solution = self.ipopt_solver.solve(
                            anchors,
                            distances,
                            ref_power=p0, # Pass randomized P0
                            ple=ple       # Pass randomized PLE
                        )
                        status_code = solution["info"].get("status", -1)
                        success = status_code in [0, 1]
                    elif solver_name == "weighted_ipopt":
                        # Calculate the exact same weights used for the SciPy weighted solver
                        weights = self._compute_weights(anchors) 
                        solution = self.ipopt_solver.solve(
                            anchors,
                            distances,
                            ref_power=p0, 
                            ple=ple,
                            weights=weights # Pass them in!
                        )
                        status_code = solution["info"].get("status", -1)
                        success = status_code in [0, 1]
                    else:
                        raise ValueError(f"Unknown solver: {solver_name}")

                    est = solution["solution"]

                    rows.append({
                        "run_id": run_id,
                        "target_id": target_id,
                        "solver": solver_name,
                        "anchor_count": self.config.anchor_count,
                        "true_x": true_position[0],
                        "true_y": true_position[1],
                        "est_x": est[0],
                        "est_y": est[1],
                        "error": euclidean_error(true_position, est),
                        "success": success,
                        # RECORD THE RANDOMIZED PARAMETERS IN THE RESULTS
                        "p0": p0,
                        "ple": ple,
                        "sigma": sigma
                    })

        return pd.DataFrame(rows)

    def _compute_weights(self, anchors):
        d = np.linalg.norm(anchors - np.mean(anchors, axis=0), axis=1)
        return 1.0 / (d + 1e-6)

    def save(self, df: pd.DataFrame, filename="results.csv"):
        df.to_csv(filename, index=False)