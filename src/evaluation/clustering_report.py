"""
clustering_report.py

Reads a clustering_results.csv produced by clustering_main.py and produces:

  1. A summary table: empirical RMSE, mean CRLB bound, and efficiency ratio
     per (scenario, solver) — lets you separate "the bound itself got worse
     under tight clustering" (bad GDOP, expected) from "the estimator fell
     further behind the bound" (an actual robustness problem for that solver
     under bad geometry).
  2. A line plot: RMSE vs cluster spacing, one line per solver, with the
     CRLB bound overlaid as a dashed reference — directly answers "how much
     does anchor clustering hurt each algorithm".
  3. A box plot per scenario: error distribution by solver, to see whether
     tighter clustering also increases variance/tail risk, not just mean error.

Usage:
    python clustering_report.py --input clustering_results.csv --outdir clustering_plots
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# A run whose position error exceeds this (metres) is treated as a
# "catastrophic" localization failure rather than a graceful degradation.
# In a 1000x1000 m area, >500 m means the estimate is essentially useless
# (worse than guessing the map centre). RMSE is dominated by these outliers,
# so we report their frequency separately — see the study notes.
CATASTROPHIC_ERROR_M = 500.0


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["scenario", "cluster_spacing", "solver"]).agg(
        empirical_rmse=("error", lambda e: np.sqrt(np.mean(np.square(e)))),
        median_error=("error", "median"),
        p90_error=("error", lambda e: np.percentile(e, 90)),
        max_error=("error", "max"),
        catastrophic_rate=("error", lambda e: np.mean(e > CATASTROPHIC_ERROR_M)),
        crlb_bound=("crlb_rmse", "mean"),
        success_rate=("success", "mean"),
        n_samples=("error", "count"),
    ).reset_index()

    # Efficiency vs the bound, reported against the MEDIAN (robust to the
    # outlier runs that inflate RMSE under bad GDOP) as well as RMSE.
    grouped["efficiency_ratio"] = grouped["crlb_bound"] / grouped["empirical_rmse"]
    grouped["median_efficiency_ratio"] = grouped["crlb_bound"] / grouped["median_error"]
    return grouped.sort_values(["cluster_spacing", "solver"])


def plot_error_vs_spacing(table: pd.DataFrame, outdir: str):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for solver, group in table.groupby("solver"):
        group = group.sort_values("cluster_spacing")
        line, = ax.plot(group["cluster_spacing"], group["empirical_rmse"], marker="o",
                         label=f"{solver} (empirical)")
        ax.plot(group["cluster_spacing"], group["crlb_bound"], linestyle="--",
                color=line.get_color(), alpha=0.6, label=f"{solver} (CRLB)")

    ax.set_xlabel("Intra-cluster anchor spacing (m)")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Localization error vs. anchor clustering tightness")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "clustering_error_vs_spacing.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_median_vs_spacing(table: pd.DataFrame, outdir: str):
    """Median error vs spacing — the robust view. Unlike RMSE (see
    plot_error_vs_spacing), the median is not dragged around by the handful of
    catastrophic bad-GDOP runs, so this is the plot that actually shows the
    'tighter clustering hurts' trend cleanly."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for solver, group in table.groupby("solver"):
        group = group.sort_values("cluster_spacing")
        line, = ax.plot(group["cluster_spacing"], group["median_error"], marker="o",
                         label=f"{solver} (median)")
        ax.plot(group["cluster_spacing"], group["crlb_bound"], linestyle="--",
                color=line.get_color(), alpha=0.5, label=f"{solver} (CRLB)")

    ax.set_xlabel("Intra-cluster anchor spacing (m)")
    ax.set_ylabel("Median error (m)")
    ax.set_title("Median localization error vs. anchor clustering tightness")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "clustering_median_vs_spacing.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_catastrophic_rate(table: pd.DataFrame, outdir: str):
    """Fraction of runs with error > CATASTROPHIC_ERROR_M, per solver per
    spacing. This is the tail-risk story RMSE hides: two solvers can share a
    median yet differ wildly in how often they blow up entirely."""
    spacings = sorted(table["cluster_spacing"].unique())
    solvers = sorted(table["solver"].unique())
    width = 0.8 / max(len(solvers), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, solver in enumerate(solvers):
        sub = table[table["solver"] == solver].set_index("cluster_spacing")
        heights = [100 * sub.loc[s, "catastrophic_rate"] if s in sub.index else 0 for s in spacings]
        positions = [x + (i - (len(solvers) - 1) / 2) * width for x in range(len(spacings))]
        ax.bar(positions, heights, width=width, label=solver)

    ax.set_xticks(range(len(spacings)))
    ax.set_xticklabels([f"{s:g} m" for s in spacings])
    ax.set_xlabel("Intra-cluster anchor spacing")
    ax.set_ylabel(f"Catastrophic failure rate (%) [error > {int(CATASTROPHIC_ERROR_M)} m]")
    ax.set_title("Tail risk vs. anchor clustering tightness")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(outdir, "clustering_catastrophic_rate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_boxplots_by_scenario(df: pd.DataFrame, outdir: str):
    scenarios = sorted(df["scenario"].unique(), key=lambda s: df[df["scenario"] == s]["cluster_spacing"].iloc[0])
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4.5), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        sub = df[df["scenario"] == scenario]
        solvers = sorted(sub["solver"].unique())
        data = [sub[sub["solver"] == s]["error"].values for s in solvers]
        ax.boxplot(data, tick_labels=solvers, showfliers=True)
        ax.set_title(scenario)
        ax.set_ylabel("Error (m)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Error distribution by solver, per clustering scenario")
    fig.tight_layout()
    path = os.path.join(outdir, "clustering_boxplots.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Anchor-clustering study report")
    parser.add_argument("--input", type=str, default="clustering_results.csv")
    parser.add_argument("--outdir", type=str, default="clustering_plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    required = {"scenario", "cluster_spacing", "crlb_rmse"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{args.input} is missing columns {missing} — it must come from clustering_main.py's "
            "ClusteringTask, not from main.py/sweep_main.py."
        )

    table = build_summary_table(df)
    table_path = os.path.join(args.outdir, "clustering_summary_table.csv")
    table.to_csv(table_path, index=False)
    print(f"Summary table written to {table_path}\n")
    print(table.to_string(index=False))

    plot_error_vs_spacing(table, args.outdir)
    plot_median_vs_spacing(table, args.outdir)
    plot_catastrophic_rate(table, args.outdir)
    plot_boxplots_by_scenario(df, args.outdir)
    print(f"\nPlots written to {args.outdir}/")


if __name__ == "__main__":
    main()
