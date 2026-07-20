"""
solver_comparison_report.py

Reads a results CSV produced by solver_comparison_main.py (or sweep_main.py/
main.py, as long as it has a `solve_time_sec` column — added by
SolverRegistry's timing wrapper) and produces:

  1. A summary table: RMSE, success rate, mean/median solve time, per solver
     (and, if crlb_rmse is present, mean efficiency ratio too) — the numeric
     evidence for "D) benchmark IPOPT against other NLP/MINLP solvers".
  2. A grouped bar chart: RMSE by solver.
  3. A grouped bar chart: mean solve time by solver (log scale — global
     solvers like SCIP can be orders of magnitude slower than a local
     interior-point method).
  4. An accuracy-vs-cost scatter ("efficiency frontier"): one point per
     solver, x = mean solve time, y = RMSE, so you can see which solvers are
     Pareto-dominated (slower AND less accurate than some alternative).

Usage:
    python solver_comparison_report.py --input solver_comparison_results.csv --outdir solver_comparison_plots
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "empirical_rmse": ("error", lambda e: np.sqrt(np.mean(np.square(e)))),
        "median_error": ("error", "median"),
        "p90_error": ("error", lambda e: np.percentile(e, 90)),
        # Robust tail metric: fraction of runs worse than 500 m (useless in a
        # 1000x1000 m area). RMSE alone hides how often a solver blows up.
        "catastrophic_rate": ("error", lambda e: np.mean(e > 500.0)),
        "success_rate": ("success", "mean"),
        "mean_solve_time_sec": ("solve_time_sec", "mean"),
        "median_solve_time_sec": ("solve_time_sec", "median"),
        "n_samples": ("error", "count"),
    }
    table = df.groupby("solver").agg(**agg).reset_index()

    if "crlb_rmse" in df.columns:
        crlb_by_solver = df.groupby("solver")["crlb_rmse"].mean()
        table["mean_crlb_bound"] = table["solver"].map(crlb_by_solver)
        table["efficiency_ratio"] = table["mean_crlb_bound"] / table["empirical_rmse"]
        table["median_efficiency_ratio"] = table["mean_crlb_bound"] / table["median_error"]

    return table.sort_values("empirical_rmse")


def plot_rmse_bar(table: pd.DataFrame, outdir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    order = table.sort_values("empirical_rmse")
    ax.bar(order["solver"], order["empirical_rmse"], color="steelblue")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Localization accuracy by solver")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(outdir, "solver_comparison_rmse.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_runtime_bar(table: pd.DataFrame, outdir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    order = table.sort_values("mean_solve_time_sec")
    ax.bar(order["solver"], order["mean_solve_time_sec"], color="darkorange")
    ax.set_yscale("log")
    ax.set_ylabel("Mean solve time (s, log scale)")
    ax.set_title("Solver cost comparison")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    path = os.path.join(outdir, "solver_comparison_runtime.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_efficiency_frontier(table: pd.DataFrame, outdir: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in table.iterrows():
        ax.scatter(row["mean_solve_time_sec"], row["empirical_rmse"], s=90)
        ax.annotate(row["solver"], (row["mean_solve_time_sec"], row["empirical_rmse"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Mean solve time (s, log scale)")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Accuracy vs. cost — lower-left is better")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "solver_comparison_frontier.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Cross-solver comparison report")
    parser.add_argument("--input", type=str, default="solver_comparison_results.csv")
    parser.add_argument("--outdir", type=str, default="solver_comparison_plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    if "solve_time_sec" not in df.columns:
        raise ValueError(
            f"{args.input} has no 'solve_time_sec' column — it must come from the updated "
            "task.py/sweep_task.py that log SolverRegistry's per-solver timing, or from "
            "solver_comparison_main.py directly."
        )

    table = build_summary_table(df)
    table_path = os.path.join(args.outdir, "solver_comparison_table.csv")
    table.to_csv(table_path, index=False)
    print(f"Summary table written to {table_path}\n")
    print(table.to_string(index=False))

    plot_rmse_bar(table, args.outdir)
    plot_runtime_bar(table, args.outdir)
    plot_efficiency_frontier(table, args.outdir)
    print(f"\nPlots written to {args.outdir}/")


if __name__ == "__main__":
    main()
