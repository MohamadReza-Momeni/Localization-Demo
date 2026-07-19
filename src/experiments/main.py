import argparse
from src.experiments.config import ExperimentConfig
from src.experiments.task import SimulationTask
from src.experiments.executor import BatchExecutor
from src.experiments.exporter import ResultExporter
from src.visualization.map_generator import LocalizationVisualizer
from src.localization.ipopt_params import IPOPTHyperparams


def main():
    parser = argparse.ArgumentParser(description="Run Localization Experiments")

    parser.add_argument("--anchors", type=int, default=6, help="Number of anchors")
    parser.add_argument("--targets", type=int, default=3, help="Number of targets")
    parser.add_argument("--runs", type=int, default=100, help="Number of experiment runs")

    parser.add_argument("--p0_min", type=float, default=-50.0)
    parser.add_argument("--p0_max", type=float, default=50.0)
    parser.add_argument("--n_min", type=float, default=2.0, help="Beta (path loss exponent) min")
    parser.add_argument("--n_max", type=float, default=8.0, help="Beta (path loss exponent) max")
    parser.add_argument("--sigma_min", type=float, default=0.0)
    parser.add_argument("--sigma_max", type=float, default=10.0)

    parser.add_argument("--het_factor", type=float, default=0.0,
                         help="Distance-dependent noise strength. 0 = homogeneous (default); "
                              ">0 makes far anchors noisier so the weighted solvers have a "
                              "reliability signal to exploit. See src/signal/rssi.py.")
    parser.add_argument("--het_reference_distance", type=float, default=100.0,
                         help="Distance (m) at which het noise adds one --het_factor of sigma.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Master RNG seed for reproducibility. Omit for fresh OS entropy "
                              "per run (the historical behaviour); set an int to make the whole "
                              "study deterministic (anchors, targets, RSSI noise).")

    parser.add_argument("--lat", type=float, default=35.7152, help="Origin Latitude")
    parser.add_argument("--lon", type=float, default=51.4043, help="Origin Longitude")

    parser.add_argument("--solvers", nargs="+",
                         default=["vanilla", "weighted", "ipopt", "weighted_ipopt", "particle_filter"],
                         help="Which solvers to evaluate")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel worker processes")
    parser.add_argument("--output", type=str, default="results.csv", help="Output CSV path")

    # --- IPOPT internal parameters (project-level, no longer hardcoded in the solver) ---
    parser.add_argument("--ipopt_tol", type=float, default=1e-6, help="IPOPT main tolerance (epsilon)")
    parser.add_argument("--ipopt_acceptable_tol", type=float, default=1e-4,
                         help="IPOPT fallback tolerance; must be >= --ipopt_tol")
    parser.add_argument("--ipopt_max_iter", type=int, default=500)
    parser.add_argument("--ipopt_hessian", type=str, default="limited-memory",
                         choices=["limited-memory", "exact"])
    parser.add_argument("--ipopt_mu_strategy", type=str, default="monotone",
                         choices=["monotone", "adaptive"])
    parser.add_argument("--ipopt_starting_points", nargs="+",
                         default=["warm_start", "anchor_centroid", "random_point"],
                         choices=["warm_start", "anchor_centroid", "random_point"],
                         help="Which candidate starting points feed IPOPT multi-start")
    parser.add_argument("--ipopt_no_multi_start", action="store_true",
                         help="Disable multi-start; only the first --ipopt_starting_points entry runs")
    parser.add_argument("--ipopt_fixed_initial_point", type=float, nargs=2, default=None,
                         metavar=("X", "Y"),
                         help="Override the warm-start candidate with an explicit (x, y), "
                              "e.g. to reproduce a MATLAB-side initial point")

    args = parser.parse_args()

    print("Starting experiments with randomized parameters...")

    ipopt_params = IPOPTHyperparams(
        tol=args.ipopt_tol,
        acceptable_tol=args.ipopt_acceptable_tol,
        max_iter=args.ipopt_max_iter,
        hessian_approximation=args.ipopt_hessian,
        mu_strategy=args.ipopt_mu_strategy,
        starting_points=tuple(args.ipopt_starting_points),
        multi_start=not args.ipopt_no_multi_start,
        fixed_initial_point=tuple(args.ipopt_fixed_initial_point) if args.ipopt_fixed_initial_point else None,
    )

    config = ExperimentConfig(
        anchor_count=args.anchors,
        target_count=args.targets,
        lat0=args.lat,
        lon0=args.lon,
        p0_range=(args.p0_min, args.p0_max),
        ple_range=(args.n_min, args.n_max),
        noise_range=(args.sigma_min, args.sigma_max),
        het_factor=args.het_factor,
        het_reference_distance=args.het_reference_distance,
        base_seed=args.seed,
        solvers=tuple(args.solvers),
        ipopt_params=ipopt_params,
    )

    task = SimulationTask(config)
    executor = BatchExecutor(task)

    results = executor.run(run_count=args.runs, max_workers=args.workers)
    ResultExporter.save_csv(results, filename=args.output)

    print("\nDONE")
    print(results.head())

    print("\nGenerating geospatial maps...")
    visualizer = LocalizationVisualizer(
        lat0=args.lat, lon0=args.lon,
        x_range=config.x_range, y_range=config.y_range,
        results_path=args.output,
    )
    visualizer.generate_run_map(run_id=0, output_filename="localization_map.html")
    print("Map saved to localization_map.html")


if __name__ == "__main__":
    main()
