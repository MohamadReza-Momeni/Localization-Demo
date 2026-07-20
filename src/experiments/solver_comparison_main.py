"""
solver_comparison_main.py

CLI entrypoint for the "D) AMPL -> Tools optimization" solver-comparison
study: run the SAME RSSI-domain NLP (identical objective, identical
multi-start candidates) across every registered solver strategy —

    vanilla, weighted            (SciPy 'trf' least-squares, distance-domain)
    ipopt, weighted_ipopt         (cyipopt, RSSI-domain, direct binding)
    ampl_ipopt                    (same IPOPT algorithm, routed through AMPL)
    ampl_bonmin                   (MINLP solver, reduces to NLP w/ 0 int vars)
    ampl_scip                     (global MINLP/NLP via spatial branch & bound)
    particle_filter

and log both accuracy (error vs CRLB, if you also pass sweep grids) and
wall-clock cost (solve_time_sec, added by SolverRegistry's timing wrapper)
for every solver on identical scenarios.

NOT included: "ampl_cbc" / "ampl_cuopt". Both are LP/MIP(/routing)-oriented
solvers that cannot represent this problem's nonlinear log-distance RSSI
objective as posed (see ampl_solver.py's SOLVER_CAPABILITIES for the
detailed reasoning per solver, and where CBC/cuOpt WOULD fit in this
project instead — anchor-placement/clustering optimization, not per-target
localization). Calling AMPLSolver().solve(solver_name="cbd") or "cuopt"
directly raises a clear ValueError rather than silently returning nonsense;
this script does not route the sweep through them.

This is a thin wrapper around the existing SweepConfig/SweepTask/
BatchExecutor pipeline (same one sweep_main.py uses) — it does not
reimplement the sweep, it just defaults --solvers to the full comparison
set and reminds you that AMPL solvers require amplpy + a registered AMPL
license (ampl.com) to actually run.

Usage:
    python solver_comparison_main.py --replications 20 --ple 2 3 4 --sigma 1 2 4 8
    python solver_comparison_main.py --solvers ipopt ampl_ipopt ampl_bonmin ampl_scip
"""
import argparse
from src.experiments.sweep_config import SweepConfig
from src.experiments.sweep_task import SweepTask
from src.experiments.executor import BatchExecutor
from src.experiments.exporter import ResultExporter
from src.localization.ipopt_params import IPOPTHyperparams
from src.localization.ampl_solver import SOLVER_CAPABILITIES

ALL_COMPARISON_SOLVERS = [
    "vanilla", "weighted", "ipopt", "weighted_ipopt",
    "ampl_ipopt", "ampl_bonmin", "ampl_scip", "particle_filter",
]


def main():
    parser = argparse.ArgumentParser(description="Cross-solver accuracy/runtime comparison")

    parser.add_argument("--anchors", type=int, default=6)
    parser.add_argument("--targets", type=int, default=1)
    parser.add_argument("--replications", type=int, default=10)

    parser.add_argument("--p0", type=float, nargs="+", default=[-40.0])
    parser.add_argument("--ple", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    parser.add_argument("--sigma", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])

    parser.add_argument("--map_width", type=float, default=1000)
    parser.add_argument("--map_height", type=float, default=1000)
    parser.add_argument("--lat", type=float, default=35.7152)
    parser.add_argument("--lon", type=float, default=51.4043)

    parser.add_argument("--solvers", nargs="+", default=ALL_COMPARISON_SOLVERS,
                         help="Which registered strategies to compare. Default: every "
                              "NLP-capable strategy (excludes ampl_cbc/ampl_cuopt, which "
                              "aren't registered — see module docstring).")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=str, default="solver_comparison_results.csv")

    parser.add_argument("--ampl_solver_options", type=str, default="",
                         help="Raw AMPL solver-options string applied to bonmin/scip/ipopt "
                              "AMPL runs, e.g. 'bonmin.algorithm=B-BB'")

    args = parser.parse_args()

    print("Solver capability notes (why cbc/cuopt aren't in this comparison):")
    for name in ("cbc", "cuopt"):
        print(f"  - {name}: {SOLVER_CAPABILITIES[name]['notes']}")
    print()

    ampl_options = {}
    if args.ampl_solver_options:
        for solver_name in ("ipopt", "bonmin", "scip"):
            if f"ampl_{solver_name}" in args.solvers:
                ampl_options[solver_name] = args.ampl_solver_options

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
        replications=args.replications,
        solvers=tuple(args.solvers),
        ipopt_params=IPOPTHyperparams(),
        ampl_options=ampl_options,
    )

    needs_ampl = any(s.startswith("ampl_") for s in args.solvers)
    if needs_ampl:
        print("NOTE: ampl_* solvers require `pip install amplpy` and a registered AMPL "
              "license with the relevant solver modules installed "
              "(`python -m amplpy.modules install <solver>`), or these rows will error out.\n")

    total_calls = config.total_solver_calls()
    print(f"Grid size per replication: {config.grid_size()} (P0 x beta x sigma)")
    print(f"Total solver evaluations: {total_calls}")
    if total_calls > 20000:
        print("WARNING: large comparison run — consider shrinking --replications or the grids.")

    task = SweepTask(config)
    executor = BatchExecutor(task)

    results = executor.run(run_count=config.replications, max_workers=args.workers)
    ResultExporter.save_csv(results, filename=args.output)

    print(f"\nDONE — {len(results)} rows written to {args.output}")
    summary = results.groupby("solver").agg(
        rmse=("error", lambda e: (e ** 2).mean() ** 0.5),
        success_rate=("success", "mean"),
        mean_solve_time_sec=("solve_time_sec", "mean"),
    )
    print(summary.sort_values("rmse").to_string())
    print(f"\nRun solver_comparison_report.py --input {args.output} for plots.")


if __name__ == "__main__":
    main()
