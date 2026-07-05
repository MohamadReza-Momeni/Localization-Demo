import argparse
from src.experiments.runner import ExperimentRunner
from src.visualization.map_generator import LocalizationVisualizer

def main():
    parser = argparse.ArgumentParser(description="Run Localization Experiments")
    
    parser.add_argument("--anchors", type=int, default=6, help="Number of anchors")
    parser.add_argument("--targets", type=int, default=3, help="Number of targets")
    parser.add_argument("--runs", type=int, default=100, help="Number of experiment runs")
    
    # Updated to ranges based on your new requirements
    parser.add_argument("--p0_min", type=float, default=-50.0)
    parser.add_argument("--p0_max", type=float, default=50.0)
    parser.add_argument("--n_min", type=float, default=2.0)
    parser.add_argument("--n_max", type=float, default=8.0)
    parser.add_argument("--sigma_min", type=float, default=0.0)
    parser.add_argument("--sigma_max", type=float, default=10.0)
    
    parser.add_argument("--lat", type=float, default=35.7152, help="Origin Latitude")
    parser.add_argument("--lon", type=float, default=51.4043, help="Origin Longitude")
    
    args = parser.parse_args()

    print("Starting experiments with randomized parameters...")

    runner = ExperimentRunner(
        anchor_count=args.anchors,
        target_count=args.targets,
        p0_range=(args.p0_min, args.p0_max),
        ple_range=(args.n_min, args.n_max),
        noise_range=(args.sigma_min, args.sigma_max),
    )

    results = runner.run_batch(run_count=args.runs)
    runner.save(results)

    print("\nDONE")
    print(results.head())

    print("\nGenerating geospatial maps...")
    visualizer = LocalizationVisualizer(lat0=args.lat, lon0=args.lon)
    visualizer.generate_run_map(run_id=0, output_filename="localization_map.html")
    print("Map saved to localization_map.html")

if __name__ == "__main__":
    main()