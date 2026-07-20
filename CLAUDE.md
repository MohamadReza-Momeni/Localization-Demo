# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

RF Localization Simulator: generates RSSI (Received Signal Strength Indicator) measurements
under a log-distance path-loss model with configurable noise, then benchmarks how accurately
different optimization strategies recover a hidden target's position — against each other and
against the theoretical Cramér-Rao Lower Bound (CRLB).

## Commands

Everything runs through the unified CLI `run.py` (forwards all args after the subcommand verbatim
to the underlying module; append `--help` to any subcommand for its options):

```bash
pip install -r requirements.txt
python -m amplpy.modules install coin   # optional: enables ampl_ipopt + ampl_bonmin (open-source, no license)

python run.py single --runs 100         # random per-run P0/beta/sigma sampling study -> results.csv + map
python run.py sweep  --ple 2 3 4 --sigma 1 2 4 8 --replications 20 --output sweep_results.csv
python run.py clustering                # anchor-clustering (GDOP) geometry study
python run.py compare --solvers ipopt ampl_ipopt ampl_bonmin --replications 20
python run.py report-crlb      --input sweep_results.csv --outdir crlb_plots
python run.py report-solvers   --input solver_comparison_results.csv
python run.py report-clustering
python run.py app                       # Streamlit dashboard (spawns `streamlit run app.py`)
python run.py tree                      # ASCII project tree (Windows cp1252 safe)
```

Studies write a CSV, then a separate `report-*` command consumes that CSV to produce plots/summary CSVs.

**Testing:** there is no project test suite or test runner configured. "Tests" in this repo means
ad-hoc verification scripts (gradient/Hessian checks) documented in `PROJECT_NOTES.md`, not a pytest suite.
Do not assume `pytest` will find anything meaningful (the only `test_*.py` files live under `.venv`).

**Platform note:** developed on Windows; shell is Git Bash. Keep console output ASCII where it matters
(see `run.py tree`).

## Architecture

The core pattern is a **Strategy registry over a shared per-run context**, reused identically across
three different study types.

**Solvers** (`src/localization/`) all subclass `BaseSolver` but there is a deliberate **domain split**:
- `scipy_solver.py` — Trust-Region-Reflective least-squares in the **linear distance domain**. Also
  serves as the warm-start seed for every run.
- `ipopt_solver.py` — `cyipopt` interior-point solver in the **logarithmic RSSI domain**, with a
  verified analytic gradient and analytic Hessian. `RSSIDomainProblem` is the cyipopt problem class.
- `ampl_solver.py` — the same RSSI-domain NLP posed once in AMPL and routed to `ipopt`/`bonmin`/`scip`.
- `particle_filter_solver.py` — probabilistic resampling solver.
- `multistart.py` — shared candidate-start generation used by BOTH the native IPOPT and AMPL paths, so
  a native-vs-AMPL comparison isolates the *solver*, not the starting point. The `random_point` start is
  **deterministically seeded from scenario geometry** so all solvers start identically and runs are reproducible.

**Solver selection** (`src/experiments/strategies.py`): `SolverRegistry` maps solver-name strings to
methods (`"vanilla"`, `"weighted"`, `"ipopt"`, `"weighted_ipopt"`, `"ampl_ipopt"`, `"ampl_bonmin"`,
`"ampl_scip"`, `"particle_filter"`). `execute_solver` wraps each call in **uniform wall-clock timing**
(measured here, once, for every solver — never inside individual solvers). To add a solver: add a
`BaseSolver` subclass, add a method + registry entry here; it becomes available everywhere automatically.

**Per-run state** flows through `RunContext` (`src/experiments/context.py`): anchors, distances, warm-start
guess, P0/PLE, map bounds, plus `ipopt_params` and `ampl_options`.

**Three study tasks**, all drop-in compatible with `BatchExecutor` (`executor.py`, a `ProcessPoolExecutor`
fan-out) — each `.execute(run_id)` returns a list of row dicts:
- `task.py` (`SimulationTask`) — random per-run (P0, beta, sigma) sampling.
- `sweep_task.py` (`SweepTask`) — one fixed anchor layout, walk the full P0×beta×sigma grid; tags each row
  with the theoretical `crlb_rmse`.
- `clustering_task.py` — anchor-geometry / GDOP study.

Each study has a matching `*_config.py` (frozen dataclass) and `*_main.py` (argparse entry point).
Results persist via `exporter.py` to CSV; `evaluation/*_report.py` scripts turn those CSVs into plots.

**IPOPT internals are a first-class, loggable parameter** (`ipopt_params.py`, `IPOPTHyperparams`):
tolerances, `hessian_approximation` (`limited-memory`/`exact`), `mu_strategy`, and the multi-start
`starting_points` set are exposed on the CLI (`--ipopt_*` flags) and written into every IPOPT-family
output row — not hardcoded in the solver. `acceptable_tol` must be `>= tol` (validated in `__post_init__`).

**Supporting modules:** `signal/rssi.py` (forward RSSI model) + `signal/distance.py` (RSSI->distance inversion);
`scenario/` (anchor/target/clustered-anchor generators, all use fresh OS entropy seeds); `evaluation/crlb.py`
(Fisher-information CRLB); `geo/enu.py` (ENU<->WGS84 via pyproj); `visualization/map_generator.py` (Folium maps).

## Domain gotchas (from PROJECT_NOTES.md / README — do not "fix" these)

- **Large errors at high noise are physical, not bugs.** Under the log-distance model a few dB of RSSI
  noise maps to large distance uncertainty, especially at low path-loss exponents. The IPOPT math is
  verified correct (analytic gradient matches numeric to ~1e-8; tracks CRLB at low noise). For clean
  results sweep sigma ~= 1–2 dB. Use `report-crlb` to confirm empirical error tracks the theoretical bound.
- **CRLB is independent of P0** by construction (P0 is an additive constant that cancels in the derivative).
  If you sweep P0 and the CRLB curve doesn't move, that's correct, not a bug.
- **The "weighted" variants add little** under the current homogeneous (same-sigma-per-anchor) noise model —
  there is nothing for reliability weighting to exploit until the noise model becomes heterogeneous.
- **AMPL solver availability:** `ampl_ipopt`/`ampl_bonmin` work after `amplpy.modules install coin`.
  `ampl_scip` needs a separately-licensed module (reports failure rather than crashing if absent).
  `ampl_cbc`/`ampl_cuopt` are intentionally NOT registered — they are LP/MIP solvers that cannot represent
  this nonlinear objective. Native solver banners are suppressed via OS-level fd redirect during batch runs.
- **IPOPT best-candidate selection** prefers a *converged* candidate (status 0/1), then lowest objective;
  `success` reflects the actually-returned candidate's convergence (it must not claim success for a
  non-converged winner).
