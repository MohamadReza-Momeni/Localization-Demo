"""
src/experiments/clustering_task.py

E) Task for the anchor-clustering geometry study. Mirrors sweep_task.py's
structure and reuses the same SolverRegistry/RunContext/RSSIModel/CRLB
machinery — only the anchor generator and the outer loop variable (scenario
instead of P0) differ.

Drop-in compatible with BatchExecutor: BatchExecutor(ClusteringTask(cfg)).run(
    run_count=cfg.replications) works exactly like it does for
SimulationTask/SweepTask, since ClusteringTask.execute(replication_id) has
the same signature/return shape.
"""
import json
import numpy as np

from src.evaluation.metrics import euclidean_error
from src.scenario.clustered_anchors import ClusteredAnchorGenerator
from src.scenario.targets import TargetGenerator
from src.signal.distance import rssi_to_distance
from src.signal.rssi import RSSIModel

from src.experiments.clustering_config import ClusteringConfig
from src.experiments.context import RunContext
from src.experiments.strategies import SolverRegistry
from src.experiments.run_support import derive_seeds, augment_ipopt_ampl_columns
from src.evaluation.crlb import crlb_rmse


class ClusteringTask:

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.registry = SolverRegistry(config.x_range, config.y_range)

    def execute(self, replication_id: int) -> list[dict]:
        """
        One replication = one fresh clustered anchor layout PER SCENARIO
        (each scenario gets its own independent draw, so scenario
        differences aren't confounded by reusing the same random layout)
        plus one fresh target, held fixed while the P0 x beta x sigma grid
        (a single point by default — see ClusteringConfig) is evaluated.
        """
        results = []

        # One target seed for the run, then a distinct (anchor, noise) seed pair
        # PER SCENARIO so each scenario's layout and noise are independent yet
        # reproducible from (base_seed, replication_id). base_seed=None keeps the
        # old fresh-entropy behaviour.
        seeds = derive_seeds(self.config.base_seed, replication_id, 1 + 2 * len(self.config.scenarios))
        target_seed = seeds[0]
        scenario_seeds = seeds[1:]

        targets = TargetGenerator(
            self.config.target_count, self.config.x_range, self.config.y_range, seed=target_seed
        ).generate()

        for scenario_idx, scenario in enumerate(self.config.scenarios):
            anchor_seed = scenario_seeds[2 * scenario_idx]
            noise_seed = scenario_seeds[2 * scenario_idx + 1]
            anchors = ClusteredAnchorGenerator(
                anchor_count=self.config.anchor_count,
                x_range=self.config.x_range,
                y_range=self.config.y_range,
                seed=anchor_seed,
                n_clusters=scenario.n_clusters,
                cluster_spacing=scenario.cluster_spacing,
            ).generate()
            anchors_json = json.dumps(anchors.tolist())
            noise_rng = np.random.default_rng(noise_seed)

            for target_id, true_position in enumerate(targets):
                for p0 in self.config.p0_values:
                    for ple in self.config.ple_values:
                        for sigma in self.config.sigma_values:
                            results.extend(
                                self._evaluate_point(
                                    replication_id, target_id, true_position,
                                    anchors, anchors_json, scenario, p0, ple, sigma, noise_rng,
                                )
                            )

        return results

    # --- HELPERS ---

    def _evaluate_point(self, replication_id, target_id, true_position,
                         anchors, anchors_json, scenario, p0, ple, sigma, noise_rng):
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

        # CRLB at this scenario's actual anchor draw — lets you check whether
        # a "worse" empirical RMSE for tight clustering is just the (correctly)
        # worse achievable bound under bad GDOP, vs. an estimator actually
        # falling further behind what's achievable (efficiency_ratio in the
        # report will separate these two effects).
        sigma_per_anchor = model.noise_std_at(anchor_dists)
        crlb_bound = crlb_rmse(anchors, true_position, ple, sigma_per_anchor)

        rows = []
        for solver_name in self.config.solvers:
            solution = self.registry.execute_solver(solver_name, ctx)
            est = solution["solution"]

            row = {
                "run_id": replication_id, "target_id": target_id, "solver": solver_name,
                "scenario": scenario.name,
                "cluster_spacing": scenario.cluster_spacing,
                "n_clusters": scenario.n_clusters,
                "anchor_count": self.config.anchor_count,
                "map_width": self.config.x_range[1],
                "map_height": self.config.y_range[1],
                "lat0": self.config.lat0,
                "lon0": self.config.lon0,
                "true_x": true_position[0], "true_y": true_position[1],
                "est_x": est[0], "est_y": est[1], "error": euclidean_error(true_position, est),
                "success": solution["success"],
                "p0": p0, "ple": ple, "sigma": sigma,
                "anchors": anchors_json,
                "crlb_rmse": crlb_bound,
                "solve_time_sec": solution["solve_time_sec"],
                # RSSI-domain objective at the returned solution (None for the
                # SciPy solvers, which optimise a different distance-domain
                # objective). Logged so native-vs-AMPL comparisons and failed-run
                # audits are possible after the fact (see PROJECT_NOTES.md).
                "objective": solution.get("objective"),
            }

            # Clustering study logs only the chosen start (hyperparameters are
            # fixed here) — hence include_ipopt_params=False.
            augment_ipopt_ampl_columns(
                row, solution, self.config.ipopt_params, include_ipopt_params=False
            )

            rows.append(row)

        return rows
