import numpy as np
import pandas as pd

from src.scenario.anchors import AnchorGenerator
from src.scenario.targets import TargetGenerator
from src.signal.rssi import RSSIModel
from src.signal.distance import rssi_to_distance
from src.localization.scipy_solver import SciPyLocalizationSolver
from src.evaluation.metrics import euclidean_error


class ExperimentRunner:

    def __init__(self,
                 N=6,
                 T=1,
                 x_range=(0, 1000),
                 y_range=(0, 1000),
                 P0=-40,
                 n=2.2,
                 sigma=2.0):

        self.N = N
        self.T = T
        self.x_range = x_range
        self.y_range = y_range

        self.model = RSSIModel(P0=P0, n=n, sigma=sigma)
        self.solver = SciPyLocalizationSolver()

    def run_single(self, run_id=0):

        anchors = AnchorGenerator(self.N, self.x_range, self.y_range).generate()
        targets = TargetGenerator(self.T, self.x_range, self.y_range).generate()

        rssi = self.model.rssi_matrix(anchors, targets)

        results = []

        for t in range(self.T):

            true = targets[t]

            rssi_vals = rssi[:, t]

            distances = np.array([
                rssi_to_distance(rssi_vals[i], self.model.P0, self.model.n)
                for i in range(self.N)
            ])

            sol = self.solver.solve(anchors, distances)

            est = sol["solution"]
            err = euclidean_error(true, est)

            results.append({
                "run_id": run_id,
                "target_id": t,
                "anchor_count": self.N,
                "true_x": true[0],
                "true_y": true[1],
                "est_x": est[0],
                "est_y": est[1],
                "error": err,
                "success": sol["success"]
            })

        return results

    def run_batch(self, L=100):

        all_results = []

        for i in range(L):
            all_results.extend(self.run_single(run_id=i))

        return pd.DataFrame(all_results)

    def save(self, df, path="results.csv"):
        df.to_csv(path, index=False)