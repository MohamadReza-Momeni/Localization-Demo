import numpy as np

from src.scenario.anchors import AnchorGenerator
from src.scenario.targets import TargetGenerator
from src.signal.rssi import RSSIModel
from src.signal.distance import rssi_to_distance
from src.localization.ipopt_solver import IPOPTSolver
from src.evaluation.metrics import euclidean_error


def main():
    # ----------------------------
    # 1. Simulation area
    # ----------------------------
    N = 6   # anchors
    T = 1   # targets

    x_range = (0, 1000)  # meters
    y_range = (0, 1000)

    # ----------------------------
    # 2. Generate anchors + targets
    # ----------------------------
    anchors = AnchorGenerator(N, x_range, y_range).generate()
    targets = TargetGenerator(T, x_range, y_range).generate()
    true_target = targets[0]

    print("\n--- TRUE TARGET ---")
    print(true_target)

    # ----------------------------
    # 3. RSSI model
    # ----------------------------
    model = RSSIModel(P0=-40, n=2.2, sigma=2.0)

    # RSSI measurements: one value per anchor for the first target.
    rssi_values = model.rssi_matrix(anchors, targets)[:, 0]

    # ----------------------------
    # 4. Convert RSSI -> distance
    # ----------------------------
    distances = np.array([
        rssi_to_distance(rssi_values[i], model.P0, model.n)
        for i in range(N)
    ])

    # ----------------------------
    # 5. Run localization
    # ----------------------------
    solver = IPOPTSolver(anchors, distances)
    result = solver.solve()
    est = result["solution"]

    # ----------------------------
    # 6. Evaluation
    # ----------------------------
    error = euclidean_error(true_target, est)

    # ----------------------------
    # 7. Output
    # ----------------------------
    print("\n--- ESTIMATED TARGET ---")
    print(est)

    print("\n--- ERROR ---")
    print(error)

    print("\n--- SOLVER INFO ---")
    print(result["info"])


if __name__ == "__main__":
    main()
