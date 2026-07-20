"""
crlb_report.py

Reads a sweep_results.csv produced by sweep_main.py (which tags every row
with its per-grid-point crlb_rmse — see sweep_task.py) and produces:

  1. A validation table: empirical RMSE vs. mean CRLB bound, per
     (solver, beta, sigma) — the numeric evidence for "algorithm validation
     against the CRLB bound" from the professor's notes.
  2. 2D plots: error vs. sigma, one line per beta, one panel per solver.
  3. 3D surface plots: (beta, sigma) -> error, one surface per solver, with
     the CRLB surface overlaid for comparison — the "3D plot -> 2D and 3D
     values" requirement.

Usage:
    python crlb_report.py --input sweep_results.csv --outdir crlb_plots
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)


def build_validation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Empirical RMSE and mean CRLB bound per (solver, ple, sigma).

    Note: CRLB is independent of P0 (see crlb.py), so grouping intentionally
    excludes p0 — this averages over whatever P0 grid you swept, which is
    the correct comparison for this model. If you later make anchor noise
    P0-dependent, this grouping needs to include p0 too.
    """
    grouped = df.groupby(["solver", "ple", "sigma"]).agg(
        empirical_rmse=("error", lambda e: np.sqrt(np.mean(np.square(e)))),
        median_error=("error", "median"),
        crlb_bound=("crlb_rmse", "mean"),
        n_samples=("error", "count"),
    ).reset_index()

    grouped["efficiency_ratio"] = grouped["crlb_bound"] / grouped["empirical_rmse"]
    # Median-based efficiency: robust to the single catastrophic runs that
    # otherwise make RMSE (and thus the RMSE-vs-beta plot) misleading — e.g. a
    # lone corner-stuck estimate can make an EASIER beta look harder. The
    # median tells the true central-tendency trend.
    grouped["median_efficiency_ratio"] = grouped["crlb_bound"] / grouped["median_error"]
    # efficiency_ratio in (0, 1]; 1.0 = estimator achieves the theoretical
    # minimum variance (fully efficient); lower = further from optimal.
    return grouped.sort_values(["solver", "ple", "sigma"])


def plot_2d_by_solver(table: pd.DataFrame, outdir: str):
    """error vs sigma, one line per beta, one subplot per solver; CRLB drawn
    as a dashed reference line per beta."""
    solvers = table["solver"].unique()
    fig, axes = plt.subplots(1, len(solvers), figsize=(5 * len(solvers), 4.5), sharey=True)
    if len(solvers) == 1:
        axes = [axes]

    for ax, solver in zip(axes, solvers):
        sub = table[table["solver"] == solver]
        for ple_val, group in sub.groupby("ple"):
            group = group.sort_values("sigma")
            line, = ax.plot(group["sigma"], group["empirical_rmse"], marker="o",
                             label=f"beta={ple_val:g} (empirical)")
            ax.plot(group["sigma"], group["crlb_bound"], linestyle="--",
                    color=line.get_color(), alpha=0.6, label=f"beta={ple_val:g} (CRLB)")
        ax.set_title(solver)
        ax.set_xlabel("sigma (noise std, dB)")
        ax.set_ylabel("RMSE (m)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Empirical RMSE vs CRLB bound, by solver")
    fig.tight_layout()
    path = os.path.join(outdir, "crlb_2d_by_solver.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_2d_median_by_solver(table: pd.DataFrame, outdir: str):
    """median error vs sigma, one line per beta, one subplot per solver; CRLB
    drawn as a dashed reference line per beta.

    This is the ROBUST companion to plot_2d_by_solver: it plots the median
    instead of RMSE, so the beta-ordering trend isn't inverted by the handful
    of catastrophic outlier runs that dominate RMSE (see build_validation_table).
    """
    solvers = table["solver"].unique()
    fig, axes = plt.subplots(1, len(solvers), figsize=(5 * len(solvers), 4.5), sharey=True)
    if len(solvers) == 1:
        axes = [axes]

    for ax, solver in zip(axes, solvers):
        sub = table[table["solver"] == solver]
        for ple_val, group in sub.groupby("ple"):
            group = group.sort_values("sigma")
            line, = ax.plot(group["sigma"], group["median_error"], marker="o",
                            label=f"beta={ple_val:g} (median)")
            ax.plot(group["sigma"], group["crlb_bound"], linestyle="--",
                    color=line.get_color(), alpha=0.6, label=f"beta={ple_val:g} (CRLB)")
        ax.set_title(solver)
        ax.set_xlabel("sigma (noise std, dB)")
        ax.set_ylabel("Median error (m)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Empirical MEDIAN error vs CRLB bound, by solver (robust to outliers)")
    fig.tight_layout()
    path = os.path.join(outdir, "crlb_2d_median_by_solver.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_3d_by_solver(table: pd.DataFrame, outdir: str):
    """(beta, sigma) -> RMSE surface, one figure per solver, with the CRLB
    surface overlaid semi-transparently for direct visual comparison."""
    paths = []
    for solver in table["solver"].unique():
        sub = table[table["solver"] == solver]
        ple_vals = np.sort(sub["ple"].unique())
        sigma_vals = np.sort(sub["sigma"].unique())

        if len(ple_vals) < 2 or len(sigma_vals) < 2:
            # A surface needs a real grid in both dimensions; skip degenerate cases.
            continue

        PLE, SIGMA = np.meshgrid(ple_vals, sigma_vals, indexing="ij")
        empirical_z = np.full_like(PLE, np.nan, dtype=float)
        crlb_z = np.full_like(PLE, np.nan, dtype=float)

        pivoted_emp = sub.pivot(index="ple", columns="sigma", values="empirical_rmse")
        pivoted_crlb = sub.pivot(index="ple", columns="sigma", values="crlb_bound")
        for i, p in enumerate(ple_vals):
            for j, s in enumerate(sigma_vals):
                if p in pivoted_emp.index and s in pivoted_emp.columns:
                    empirical_z[i, j] = pivoted_emp.loc[p, s]
                    crlb_z[i, j] = pivoted_crlb.loc[p, s]

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(PLE, SIGMA, empirical_z, cmap="viridis", alpha=0.85,
                         label="empirical RMSE")
        ax.plot_surface(PLE, SIGMA, crlb_z, color="red", alpha=0.25,
                         label="CRLB bound")
        ax.set_xlabel("beta (path-loss exponent)")
        ax.set_ylabel("sigma (noise std, dB)")
        ax.set_zlabel("RMSE (m)")
        ax.set_title(f"{solver}: empirical RMSE (solid) vs CRLB (translucent red)")

        path = os.path.join(outdir, f"crlb_3d_{solver}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def main():
    parser = argparse.ArgumentParser(description="CRLB validation report from sweep results")
    parser.add_argument("--input", type=str, default="sweep_results.csv")
    parser.add_argument("--outdir", type=str, default="crlb_plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    if "crlb_rmse" not in df.columns:
        raise ValueError(
            f"{args.input} has no 'crlb_rmse' column — it must come from sweep_main.py's "
            "SweepTask, which tags every row with its per-grid-point CRLB bound. "
            "Random-sampling runs from main.py do not have this column."
        )

    table = build_validation_table(df)
    table_path = os.path.join(args.outdir, "crlb_validation_table.csv")
    table.to_csv(table_path, index=False)
    print(f"Validation table written to {table_path}\n")
    print(table.to_string(index=False))

    plot_2d_by_solver(table, args.outdir)
    plot_2d_median_by_solver(table, args.outdir)
    plot_3d_by_solver(table, args.outdir)
    print(f"\nPlots written to {args.outdir}/")


if __name__ == "__main__":
    main()
