from src.experiments.runner import ExperimentRunner
from src.visualization.map_generator import LocalizationVisualizer

ANCHOR_COUNT = 6
TARGET_COUNT = 3
RUN_COUNT = 100
NOISE_STD = 2.0
PATH_LOSS_EXPONENT = 2.2
ORIGIN_LAT = 35.7152
ORIGIN_LON = 51.4043

def main():
    runner = ExperimentRunner(
        anchor_count=ANCHOR_COUNT,
        target_count=TARGET_COUNT,
        noise_std=NOISE_STD,
        path_loss_exponent=PATH_LOSS_EXPONENT,
    )

    print("Running batch experiments...")

    results = runner.run_batch(run_count=RUN_COUNT)

    runner.save(results)

    print("\nDONE")
    print(results.head())

    print("\nSTATS:")
    print(results["error"].describe())

    print("\nGenerating geospatial maps...")
    visualizer = LocalizationVisualizer(lat0=ORIGIN_LAT, lon0=ORIGIN_LON)
    visualizer.generate_run_map(run_id=0, output_filename="localization_map.html")

if __name__ == "__main__":
    main()
