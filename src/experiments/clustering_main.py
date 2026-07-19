"""
clustering_main.py

CLI entrypoint for the "E) 3 scenarios, 20m-40m clustering" anchor-geometry
study. Runs the tight/medium/loose intra-cluster-spacing scenarios (see
clustering_config.py's DEFAULT_SCENARIOS) across `replications` independent
anchor-layout + target draws each, with P0/beta/sigma held fixed by default
so error differences are attributable to geometry, not signal conditions.

Example:
    python clustering_main.py --replications 30 --spacings 20 30 40 --n_clusters 2
"""
import argparse
from src.experiments.clustering_config import ClusteringConfig, ClusteringScenario
from src.experiments.clustering_task import ClusteringTask
from src.experiments.executor import BatchExecutor
from src.experiments.exporter import ResultExporter
from src.localization.ipopt_params import IPOPTHyperparams


def main():
    parser = argparse.ArgumentParser(description="Run the anchor-clustering geometry study")

    parser.add_argument("--anchors", type=int, default=6)
    parser.add_argument("--targets", type=int, default=1)
    parser.add_argument("--replications", type=int, default=30)

    parser.add_argument("--spacings", type=float, nargs="+", default=[20.0, 30.0, 40.0],
                         help="Intra-cluster anchor spacing per scenario, in meters")
    parser.add_argument("--n_clusters", type=int, nargs="+", default=None,
                         help="Number of clusters per scenario (default: 2 for every scenario). "
                              "If given, must match --spacings in length.")

    parser.add_argument("--p0", type=float, nargs="+", default=[-40.0])
    parser.add_argument("--ple", type=float, nargs="+", default=[3.0])
    parser.add_argument("--sigma", type=float, nargs="+", default=[2.0])

    parser.add_argument("--het_factor", type=float, default=0.0,
                         help="Distance-dependent noise strength (0 = homogeneous). See rssi.py.")
    parser.add_argument("--het_reference_distance", type=float, default=100.0,
                         help="Distance (m) at which het noise adds one --het_factor of sigma.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Master RNG seed; omit for OS entropy, set for reproducible runs.")

    parser.add_argument("--map_width", type=float, default=1000)
    parser.add_argument("--map_height", type=float, default=1000)
    parser.add_argument("--lat", type=float, default=35.7152)
    parser.add_argument("--lon", type=float, default=51.4043)

    parser.add_argument("--solvers", nargs="+",
                         default=["vanilla", "weighted", "ipopt", "weighted_ipopt", "particle_filter"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=str, default="clustering_results.csv")

    args = parser.parse_args()

    n_clusters_list = args.n_clusters or [2] * len(args.spacings)
    if len(n_clusters_list) != len(args.spacings):
        raise ValueError("--n_clusters must have the same length as --spacings if provided")

    scenarios = tuple(
        ClusteringScenario(
            name=f"spacing_{spacing:g}m",
            cluster_spacing=spacing,
            n_clusters=n_clusters,
        )
        for spacing, n_clusters in zip(args.spacings, n_clusters_list)
    )

    config = ClusteringConfig(
        anchor_count=args.anchors,
        target_count=args.targets,
        x_range=(0, args.map_width),
        y_range=(0, args.map_height),
        lat0=args.lat,
        lon0=args.lon,
        scenarios=scenarios,
        p0_values=tuple(args.p0),
        ple_values=tuple(args.ple),
        sigma_values=tuple(args.sigma),
        het_factor=args.het_factor,
        het_reference_distance=args.het_reference_distance,
        base_seed=args.seed,
        replications=args.replications,
        solvers=tuple(args.solvers),
        ipopt_params=IPOPTHyperparams(),
    )

    total_calls = config.total_solver_calls()
    print(f"Scenarios: {[s.name for s in scenarios]}")
    print(f"Signal grid size per scenario per replication: {config.grid_size()}")
    print(f"Total solver evaluations: {total_calls} "
          f"({config.replications} replications x {len(scenarios)} scenarios "
          f"x {config.target_count} targets x {config.grid_size()} signal points "
          f"x {len(config.solvers)} solvers)")
    if total_calls > 20000:
        print("WARNING: large run — consider shrinking --replications.")

    task = ClusteringTask(config)
    executor = BatchExecutor(task)

    results = executor.run(run_count=config.replications, max_workers=args.workers)
    ResultExporter.save_csv(results, filename=args.output)

    print(f"\nDONE — {len(results)} rows written to {args.output}")
    print(results.groupby(["scenario", "solver"])["error"].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
