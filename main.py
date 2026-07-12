import argparse
import pandas as pd
from src.experiments.config import ExperimentConfig
from src.experiments.task import SimulationTask
from src.experiments.executor import BatchExecutor
from src.experiments.exporter import ResultExporter
from src.visualization.map_generator import LocalizationVisualizer


def main():
    parser = argparse.ArgumentParser(description="Run Localization Experiments")

    parser.add_argument("--anchors", type=int, default=6, help="Number of anchors")
    parser.add_argument("--targets", type=int, default=3, help="Number of targets")
    parser.add_argument("--runs", type=int, default=100, help="Number of experiment runs")

    # Realistic RSSI/path-loss ranges (tightened from the original -50..50 / 2..8 / 0..10)
    parser.add_argument("--p0_min", type=float, default=-50.0)
    parser.add_argument("--p0_max", type=float, default=-30.0)
    parser.add_argument("--n_min", type=float, default=2.0)
    parser.add_argument("--n_max", type=float, default=4.0)
    parser.add_argument("--sigma_min", type=float, default=1.0)
    parser.add_argument("--sigma_max", type=float, default=3.0)

    parser.add_argument("--lat", type=float, default=35.7152, help="Origin Latitude")
    parser.add_argument("--lon", type=float, default=51.4043, help="Origin Longitude")

    args = parser.parse_args()

    print("Starting experiments with randomized parameters...")

    config = ExperimentConfig(
        anchor_count=args.anchors,
        target_count=args.targets,
        p0_range=(args.p0_min, args.p0_max),
        ple_range=(args.n_min, args.n_max),
        noise_range=(args.sigma_min, args.sigma_max),
    )

    task = SimulationTask(config)
    executor = BatchExecutor(task)
    results = executor.run(run_count=args.runs)
    ResultExporter.save_csv(results, filename="results.csv")

    print("\nDONE")

    # Widen the console display so numeric columns don't get truncated with "...",
    # and drop the long "anchors" JSON column from the preview only — it's still
    # fully saved in results.csv, it's just too wide/noisy to dump into the terminal.
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", 40,
    ):
        preview_cols = [c for c in results.columns if c != "anchors"]
        print(results[preview_cols].head())

    print(f"\nFull results (including per-run anchor coordinates) saved to results.csv")
    print(f"Total rows: {len(results)}")

    print("\nGenerating geospatial maps...")
    visualizer = LocalizationVisualizer(
        lat0=args.lat, lon0=args.lon,
        x_range=config.x_range, y_range=config.y_range,
    )
    visualizer.generate_run_map(run_id=0, output_filename="localization_map.html")
    print("Map saved to localization_map.html")


if __name__ == "__main__":
    main()