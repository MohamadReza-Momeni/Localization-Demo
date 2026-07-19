"""
sweep_main.py

CLI entrypoint for the structured P0 x beta x sigma grid sweep (as opposed
to main.py's random per-run sampling). Reuses BatchExecutor/ResultExporter
unchanged since SweepTask.execute(replication_id) matches the same
(int) -> list[dict] interface SimulationTask.execute(run_id) does.

Example:
    python sweep_main.py --p0 -40 --ple 2 3 4 --sigma 1 2 4 8 --replications 20
"""
import argparse
from src.experiments.sweep_config import SweepConfig
from src.experiments.sweep_task import SweepTask
from src.experiments.executor import BatchExecutor
from src.experiments.exporter import ResultExporter
from src.localization.ipopt_params import IPOPTHyperparams


def main():
    parser = argparse.ArgumentParser(description="Run a structured P0/beta/sigma grid sweep")

    parser.add_argument("--anchors", type=int, default=6)
    parser.add_argument("--targets", type=int, default=1)
    parser.add_argument("--replications", type=int, default=10,
                         help="Independent anchor/target draws; the full grid runs on each")

    parser.add_argument("--p0", type=float, nargs="+", default=[-40.0],
                         help="P0 grid values (dBm)")
    parser.add_argument("--ple", type=float, nargs="+", default=[2.0, 3.0, 4.0],
                         help="Beta / path-loss-exponent grid values")
    parser.add_argument("--sigma", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0],
                         help="Noise std-dev grid values")

    parser.add_argument("--het_factor", type=float, default=0.0,
                         help="Distance-dependent noise strength (0 = homogeneous). See rssi.py.")
    parser.add_argument("--het_reference_distance", type=float, default=100.0,
                         help="Distance (m) at which het noise adds one --het_factor of sigma.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Master RNG seed; omit for OS entropy, set for reproducible sweeps.")

    parser.add_argument("--map_width", type=float, default=1000)
    parser.add_argument("--map_height", type=float, default=1000)
    parser.add_argument("--lat", type=float, default=35.7152)
    parser.add_argument("--lon", type=float, default=51.4043)

    parser.add_argument("--solvers", nargs="+",
                         default=["vanilla", "weighted", "ipopt", "weighted_ipopt", "particle_filter"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=str, default="sweep_results.csv")

    parser.add_argument("--ipopt_tol", type=float, default=1e-6)
    parser.add_argument("--ipopt_acceptable_tol", type=float, default=1e-4)
    parser.add_argument("--ipopt_max_iter", type=int, default=500)
    parser.add_argument("--ipopt_hessian", type=str, default="limited-memory",
                         choices=["limited-memory", "exact"])
    parser.add_argument("--ipopt_mu_strategy", type=str, default="monotone",
                         choices=["monotone", "adaptive"])

    args = parser.parse_args()

    ipopt_params = IPOPTHyperparams(
        tol=args.ipopt_tol,
        acceptable_tol=args.ipopt_acceptable_tol,
        max_iter=args.ipopt_max_iter,
        hessian_approximation=args.ipopt_hessian,
        mu_strategy=args.ipopt_mu_strategy,
    )

    config = SweepConfig(
        anchor_count=args.anchors,
        target_count=args.targets,
        x_range=(0, args.map_width),
        y_range=(0, args.map_height),
        lat0=args.lat,
        lon0=args.lon,
        p0_values=tuple(args.p0),
        ple_values=tuple(args.ple),
        sigma_values=tuple(args.sigma),
        het_factor=args.het_factor,
        het_reference_distance=args.het_reference_distance,
        base_seed=args.seed,
        replications=args.replications,
        solvers=tuple(args.solvers),
        ipopt_params=ipopt_params,
    )

    total_calls = config.total_solver_calls()
    print(f"Grid size per replication: {config.grid_size()} (P0 x beta x sigma)")
    print(f"Total solver evaluations: {total_calls} "
          f"({config.replications} replications x {config.target_count} targets "
          f"x {config.grid_size()} grid points x {len(config.solvers)} solvers)")
    if total_calls > 20000:
        print("WARNING: this is a large sweep, especially with IPOPT multi-start enabled. "
              "Consider shrinking --replications or the grids, or disabling multi-start.")

    task = SweepTask(config)
    executor = BatchExecutor(task)

    results = executor.run(run_count=config.replications, max_workers=args.workers)
    ResultExporter.save_csv(results, filename=args.output)

    print(f"\nDONE — {len(results)} rows written to {args.output}")
    print(results.groupby(["solver", "p0", "ple", "sigma"])["error"].mean().head(20))


if __name__ == "__main__":
    main()
