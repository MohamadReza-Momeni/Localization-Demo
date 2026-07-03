from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.metrics import euclidean_error
from src.localization.scipy_solver import SciPyLocalizationSolver
from src.scenario.anchors import AnchorGenerator
from src.scenario.targets import TargetGenerator
from src.signal.distance import rssi_to_distance
from src.signal.rssi import RSSIModel


@dataclass(frozen=True)
class ExperimentConfig:
    anchor_count: int = 6
    target_count: int = 1
    x_range: tuple[float, float] = (0, 1000)
    y_range: tuple[float, float] = (0, 1000)

    reference_power: float = -40
    path_loss_exponent: float = 2.2
    noise_std: float = 2.0

    solvers: tuple[str, ...] = ("vanilla", "weighted")


class ExperimentRunner:
    def __init__(
        self,
        anchor_count=6,
        target_count=1,
        x_range=(0, 1000),
        y_range=(0, 1000),
        reference_power=-40,
        path_loss_exponent=2.2,
        noise_std=2.0,
    ):
        self.config = ExperimentConfig(
            anchor_count=anchor_count,
            target_count=target_count,
            x_range=x_range,
            y_range=y_range,
            reference_power=reference_power,
            path_loss_exponent=path_loss_exponent,
            noise_std=noise_std,
        )

        self.model = RSSIModel(
            reference_power=reference_power,
            path_loss_exponent=path_loss_exponent,
            noise_std=noise_std,
        )
        self.solver = SciPyLocalizationSolver()

    def run_single(self, run_id=0):
        anchors, targets = self._generate_scenario(run_id)
        rssi_matrix = self.model.rssi_matrix(anchors, targets)

        rows = []

        for target_id, target in enumerate(targets):
            rows.extend(
                self._localize_target(
                    run_id,
                    target_id,
                    target,
                    anchors,
                    rssi_matrix[:, target_id],
                )
            )

        return rows

    def run_batch(self, run_count=100):
        rows = []

        for run_id in range(run_count):
            rows.extend(self.run_single(run_id=run_id))

        return pd.DataFrame(rows)

    def save(self, results, path="results.csv"):
        try:
            results.to_csv(path, index=False)
        except PermissionError as exc:
            raise PermissionError(
                f"Could not write '{path}'. Close the file if it is open and try again."
            ) from exc

    def _generate_scenario(self, run_id):
        anchor_seed = 42 + run_id
        target_seed = 1 + run_id

        anchors = AnchorGenerator(
            self.config.anchor_count,
            self.config.x_range,
            self.config.y_range,
            seed=anchor_seed,
        ).generate()
        targets = TargetGenerator(
            self.config.target_count,
            self.config.x_range,
            self.config.y_range,
            seed=target_seed,
        ).generate()

        return anchors, targets

    def _localize_target(
        self,
        run_id,
        target_id,
        true_position,
        anchors,
        rssi_values,
    ):
        distances = self._estimate_distances(rssi_values)

        rows = []

        for solver_name in self.config.solvers:

            if solver_name == "vanilla":
                solution = self.solver.solve(anchors, distances)

            elif solver_name == "weighted":
                weights = self._compute_weights(anchors)
                solution = self.solver.solve(
                    anchors,
                    distances,
                    weights=weights,
                )

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
                "success": solution["success"],
            })

        return rows

    def _estimate_distances(self, rssi_values):
        return np.array([
            rssi_to_distance(
                rssi,
                self.model.reference_power,
                self.model.path_loss_exponent,
            )
            for rssi in rssi_values
        ])
    def _compute_weights(self, anchors):
        d = np.linalg.norm(anchors - np.mean(anchors, axis=0), axis=1)
        return 1.0 / (d**2 + 1e-6)
