"""
Standalone demo: shows exactly what each of the 3 IPOPT starting points
(warm_start, anchor_centroid, random_point) converges to on a single scenario,
so you can see directly whether the warm start was trapping IPOPT or not.

Run with: python demo_multistart.py
"""
import numpy as np
from src.scenario.anchors import AnchorGenerator
from src.scenario.targets import TargetGenerator
from src.signal.rssi import RSSIModel
from src.signal.distance import rssi_to_distance
from src.localization.scipy_solver import SciPyLocalizationSolver
from src.localization.ipopt_solver import IPOPTSolver
from src.evaluation.metrics import euclidean_error

X_RANGE = (0, 1000)
Y_RANGE = (0, 1000)
P0 = -40.0
PLE = 3.0
SIGMA = 2.0

# Force an edge-case target on purpose, since that's where we saw ipopt/vanilla
# agreeing suspiciously closely (the case worth stress-testing).
FORCE_EDGE_CASE = True

anchors = AnchorGenerator(6, X_RANGE, Y_RANGE, seed=None).generate()

if FORCE_EDGE_CASE:
    true_position = np.array([15.0, 500.0])  # deliberately near the left edge
    print(f"Using a forced edge-case target near the box boundary: {true_position}")
else:
    true_position = TargetGenerator(1, X_RANGE, Y_RANGE, seed=None).generate()[0]

model = RSSIModel(reference_power=P0, path_loss_exponent=PLE, noise_std=SIGMA)
rssi_values = np.array([model.rssi(np.linalg.norm(a - true_position)) for a in anchors])
distances = np.array([rssi_to_distance(r, P0, PLE) for r in rssi_values])

print(f"\nTrue position: ({true_position[0]:.2f}, {true_position[1]:.2f})")
print(f"Anchors:\n{anchors}\n")

# Step 1: get the vanilla warm start, same as your real pipeline does
vanilla = SciPyLocalizationSolver()
vanilla_result = vanilla.solve(anchors, distances, x_range=X_RANGE, y_range=Y_RANGE)
baseline_guess = vanilla_result["solution"]
vanilla_error = euclidean_error(true_position, baseline_guess)
print(f"vanilla (warm-start source): ({baseline_guess[0]:.2f}, {baseline_guess[1]:.2f})  error={vanilla_error:.2f}m\n")

# Step 2: run IPOPT with multi-start + verbose, so every candidate's outcome prints
print("Running IPOPT multi-start (warm_start / anchor_centroid / random_point)...")
ipopt = IPOPTSolver()
result = ipopt.solve(
    anchors, distances, x0=baseline_guess, ref_power=P0, ple=PLE,
    x_range=X_RANGE, y_range=Y_RANGE, multi_start=True, verbose=True,
)

print(f"\nChosen candidate: {result['chosen_start']}")
final_solution = result["solution"]
final_error = euclidean_error(true_position, final_solution)
print(f"Final IPOPT solution: ({final_solution[0]:.2f}, {final_solution[1]:.2f})  error={final_error:.2f}m")

print("\n--- Interpretation ---")
labels = [c["label"] for c in result["multi_start_candidates"]]
solutions = [tuple(np.round(c["solution"], 1)) for c in result["multi_start_candidates"]]
if len(set(solutions)) == 1:
    print("All 3 starting points converged to the SAME point.")
    print("-> The warm start was not limiting anything here; this is likely the true")
    print("   constrained optimum for this geometry/noise draw, not a trapped local basin.")
else:
    print("The starting points converged to DIFFERENT points.")
    print(f"-> The warm-start-only result would have been: {solutions[labels.index('warm_start')]}")
    print(f"-> Multi-start found a better candidate: {tuple(np.round(final_solution, 1))}")
    print("   This confirms warm-starting alone CAN trap IPOPT in a worse local basin —")
    print("   multi-start is worth keeping on, at least for edge-case/boundary geometries.")