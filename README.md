# RF Localization Simulator

A modular Python framework for simulating, benchmarking, and visualizing
RF-based (Radio Frequency) localization algorithms. It generates Received
Signal Strength Indicator (RSSI) measurements under a log-distance path-loss
model with configurable noise, then compares how accurately different
optimization strategies recover a hidden target's position — against each
other and against the theoretical Cramér-Rao Lower Bound (CRLB).

## Features

* **Multiple solver families** behind a clean Strategy-Pattern registry:
  SciPy least-squares, native IPOPT (via `cyipopt`), AMPL-routed solvers
  (IPOPT / BonMin / SCIP), and a particle filter.
* **Three complementary studies:** random Monte-Carlo sampling, a structured
  `P0 × β × σ` grid sweep, and an anchor-clustering (GDOP) geometry study.
* **CRLB validation:** every grid point is tagged with its theoretical error
  bound so empirical RMSE can be compared to the best any unbiased estimator
  could achieve.
* **Parallel processing:** batches fan out across CPU cores via
  `ProcessPoolExecutor`.
* **Interactive dashboard** (Streamlit + Folium) and **static reports**
  (matplotlib 2D/3D plots + summary CSVs).
* **Geospatial mapping:** ENU ↔ WGS84 projection (`pyproj`) renders results
  onto real-world maps.

## Quick Start

```bash
# 1. create and activate a virtualenv, then:
pip install -r requirements.txt

# 2. (optional) enable the AMPL solvers — open-source, no license needed:
python -m amplpy.modules install coin      # provides ampl_ipopt + ampl_bonmin

# 3. run something
python run.py single --runs 100            # random-sampling study
python run.py app                          # interactive dashboard
```

Everything is driven through a single entry point, `run.py`:

| Command | What it does |
|---|---|
| `python run.py single`            | Random per-run `P0/β/σ` sampling study (writes `results.csv` + a map) |
| `python run.py sweep`             | Structured `P0 × β × σ` grid sweep with CRLB |
| `python run.py clustering`        | Anchor-clustering geometry (GDOP) study |
| `python run.py compare`           | Cross-solver accuracy vs. runtime comparison |
| `python run.py report-crlb`       | Empirical RMSE vs. CRLB report (2D + 3D plots) |
| `python run.py report-solvers`    | Solver accuracy/runtime/efficiency-frontier report |
| `python run.py report-clustering` | Clustering study report |
| `python run.py app`               | Launch the Streamlit dashboard |
| `python run.py tree`              | Print a clean project tree |

Pass `--help` after any subcommand to see its options, e.g.
`python run.py sweep --help`. All args are forwarded verbatim:

```bash
python run.py sweep   --ple 2 3 4 --sigma 1 2 4 8 --replications 20 --output sweep_results.csv
python run.py report-crlb --input sweep_results.csv --outdir crlb_plots

python run.py compare --solvers ipopt ampl_ipopt ampl_bonmin --replications 20
python run.py report-solvers --input solver_comparison_results.csv
```

## Project Structure

```text
Localization-Demo/
├── run.py                          # Unified CLI entry point
├── app.py                          # Streamlit dashboard
├── requirements.txt
├── src/
│   ├── experiments/                # Orchestration
│   │   ├── main.py                 # random-sampling study
│   │   ├── sweep_config.py / sweep_task.py / sweep_main.py       # P0×β×σ grid
│   │   ├── clustering_config.py / clustering_task.py / clustering_main.py  # GDOP study
│   │   ├── solver_comparison_main.py                             # cross-solver study
│   │   ├── config.py / context.py / task.py                     # config + per-run state
│   │   ├── strategies.py           # Strategy registry + uniform timing wrapper
│   │   ├── executor.py             # parallel batch runner
│   │   └── exporter.py             # CSV persistence
│   ├── localization/               # The solvers
│   │   ├── base_solver.py
│   │   ├── scipy_solver.py         # Trust-Region-Reflective least-squares (distance domain)
│   │   ├── ipopt_solver.py         # cyipopt interior-point (RSSI/log domain, analytic gradient)
│   │   ├── ipopt_params.py         # IPOPT hyperparameters (tol, multi-start, ...)
│   │   ├── ampl_solver.py          # AMPL-routed ipopt/bonmin/scip
│   │   ├── multistart.py           # shared multi-start candidate generation
│   │   └── particle_filter_solver.py
│   ├── scenario/                   # anchors, clustered anchors, targets
│   ├── signal/                     # RSSI forward model + distance inversion
│   ├── evaluation/                 # metrics, CRLB, report generators
│   ├── geo/                        # ENU ↔ WGS84 conversion
│   └── visualization/              # Folium map generation
```

## Implemented Algorithms

1. **Vanilla (SciPy):** Trust-Region-Reflective least-squares in the linear
   distance domain. Fast, bounded, used as the warm-start seed for IPOPT.
2. **Weighted (SciPy):** adds inverse-distance geometric weighting.
3. **IPOPT:** interior-point optimizer in the logarithmic RSSI domain, with a
   verified analytic gradient and a multi-start strategy (warm-start /
   anchor-centroid / random) that keeps the lowest-objective solution.
4. **Weighted IPOPT:** IPOPT with RSSI-domain reliability weighting.
5. **AMPL solvers** (`ampl_ipopt`, `ampl_bonmin`, `ampl_scip`): the same NLP
   posed once in AMPL and routed to any registered solver, to separate the
   *solver* from the *interface* (see notes below).
6. **Particle Filter:** probabilistic resampling solver, resilient to the
   non-linear noise that traps gradient-based methods.

## Notes on accuracy (important)

* **The IPOPT math is correct.** Its analytic gradient matches a numerical
  gradient to ~1e-8, and at low noise it tracks the CRLB closely.
* **Large errors at high noise are physical, not a bug.** Under the
  log-distance model, a few dB of RSSI noise maps to a *large* distance
  uncertainty, especially at low path-loss exponents. Sweeps that average over
  `σ` up to 8 dB will show RMSE in the hundreds of metres in a 1000×1000 m
  area — the CRLB is correspondingly large there too. Use `report-crlb` to
  confirm empirical error is tracking the theoretical bound. For "clean"
  results, sweep `σ ≈ 1–2 dB`.

## Notes on the AMPL solvers

* `ampl_ipopt` and `ampl_bonmin` work out of the box after
  `python -m amplpy.modules install coin` (open-source, no license). They
  converge to the same objective as native IPOPT, validating the interface.
* `ampl_scip` is **not** in the free `coin` bundle — it needs a separately
  licensed AMPL SCIP module. Without it, `ampl_scip` rows report failure
  rather than crashing.
* `ampl_cbc` / `ampl_cuopt` are intentionally **not** registered: they are
  LP/MIP solvers that cannot represent this nonlinear objective.
* Native solver banners are suppressed automatically during batch runs.

## Adding a New Solver

1. Create `src/localization/your_solver.py` subclassing `BaseSolver`.
2. Register it in `src/experiments/strategies.py`.
3. It is automatically available to the CLI, dashboard, and parallel runner.
