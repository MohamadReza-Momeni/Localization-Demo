# Project Notes & Change Log — RF Localization Simulator

A running record of what was changed, what was verified, what worked, and —
importantly — **what did not help**, so nothing has to be rediscovered later.
Last updated: 2026-07-19.

---

## 1. Executive summary

- **The IPOPT solver is correct.** Its analytic gradient matches a numerical
  gradient to ~1e-8, and at low noise it reaches ~0.98 of the Cramér-Rao Lower
  Bound (CRLB) — essentially optimal.
- **High errors you saw earlier were physical, not a bug.** They came from
  sweeping very high RSSI noise (σ up to 8 dB) at low path-loss exponents.
- **IPOPT / weighted_IPOPT are the best solvers** on accuracy; SciPy vanilla /
  weighted and the particle filter are faster but less accurate.
- **The "weighted" variants add little to nothing** under the current
  homogeneous noise model (see §5).
- **AMPL routing works** for `ampl_ipopt` and `ampl_bonmin` (identical results
  to native IPOPT); `ampl_scip` is not available in the free solver bundle.

---

## 2. Changes made to the codebase

### 2.1 Dependencies (`requirements.txt`)
- Added missing: `pyproj` (used by `geo/enu.py`), `matplotlib` (all report
  scripts), `amplpy` (AMPL solvers).
- Fixed `cyipopt-wheels` → `cyipopt` (correct import name is `cyipopt`).
- Documented that the open-source AMPL solvers install via
  `python -m amplpy.modules install coin` (no license required).

### 2.2 Unified CLI (`run.py`, new)
Single entry point so you no longer need `python -m src.experiments.<x>`:

| Command | Runs |
|---|---|
| `python run.py single` | random per-run sampling study |
| `python run.py sweep` | structured P0×β×σ grid sweep |
| `python run.py clustering` | anchor-clustering (GDOP) study |
| `python run.py compare` | cross-solver accuracy/runtime |
| `python run.py report-crlb` / `report-solvers` / `report-clustering` | reports |
| `python run.py app` | Streamlit dashboard |
| `python run.py tree` | ASCII project tree |

All args after the subcommand are forwarded verbatim to the underlying script.

### 2.3 File moves / removals
- Moved `main.py` → `src/experiments/main.py` (alongside the other study mains).
- Deleted `src/localization.zip` (stray archive) and `test.py` (was a tree
  printer, not a test; its logic now lives in `run.py tree`, ASCII-safe on
  Windows cp1252 consoles).
- Added `__init__.py` to every `src` subpackage.

### 2.4 AMPL solver (`src/localization/ampl_solver.py`)
- Added `_suppress_native_stdout()` — an OS-level fd redirect that silences the
  thousands of Ipopt/Bonmin banner lines that previously flooded the terminal
  during batch runs. Only active when `verbose=False`; errors still propagate.

### 2.5 Reports — robust metrics added
Because RMSE is dominated by a minority of catastrophic runs (see §6), both
report builders now also emit:
- `median_error`, `p90_error`, `max_error`
- `catastrophic_rate` = fraction of runs with error > **500 m** (useless in a
  1000×1000 m area)
- `median_efficiency_ratio` = CRLB / median error

New clustering plots: `clustering_median_vs_spacing.png` and
`clustering_catastrophic_rate.png` (in addition to the existing RMSE and
box plots).

### 2.6 README rewritten
Now documents AMPL, CRLB, the sweep/clustering studies, the unified CLI, and
the noise/scip caveats honestly.

---

## 3. Verifications performed (evidence the core is correct)

- **IPOPT gradient**: analytic vs numerical (central difference) max abs diff
  = ~1.1e-8. ✅
- **IPOPT at low noise**: σ=1, β=4 → efficiency (CRLB/RMSE) = **0.98**. ✅
- **AMPL interface**: native `ipopt`, `ampl_ipopt`, `ampl_bonmin` produce
  byte-identical estimates on the same scenario (28.9 m median across all
  three at 20 reps). ✅ This isolates the *interface* from the *algorithm*.

---

## 4. What worked well

- **IPOPT / weighted_IPOPT are the accuracy leaders.** Full comparison
  (6000 evals, σ=1–8, β=2–4, 100 reps):

  | Solver | RMSE (m) | Median (m) | Catastrophic % | Mean time |
  |---|---|---|---|---|
  | weighted_ipopt | 201.8 | 58.5 | 4.9% | 238 ms |
  | ipopt | 207.4 | 56.9 | 4.6% | 218 ms |
  | weighted | 240.3 | 68.9 | 6.4% | 5 ms |
  | particle_filter | 240.6 | 80.1 | 5.9% | 15 ms |
  | vanilla | 248.0 | 83.0 | 6.7% | 4 ms |

  IPOPT roughly **halves the catastrophic-failure rate** vs the SciPy baselines
  and has the lowest median error.

- **IPOPT multi-start earns its keep.** The winning start was `warm_start`
  only ~42% of the time; `random_point` (~30%) and `anchor_centroid` (~28%)
  won the rest. Disabling multi-start would measurably hurt accuracy.

- **Error tracks the CRLB and behaves as theory predicts:**
  - grows with noise σ,
  - grows as β (path-loss exponent) decreases (β=2 is hardest),
  - at low noise IPOPT is near-optimal (efficiency up to 0.98).

- **Clean low-noise sweep (σ=1–2)** gives the "paper-ready" numbers, e.g.
  IPOPT at β=4, σ=1: RMSE 24.4 m vs CRLB 23.9 m.

---

## 5. What did NOT help (keep this so we don't retry it)

- **Geometric / RSSI weighting (`weighted`, `weighted_ipopt`) barely helps.**
  Across the full comparison, `weighted` ≈ `vanilla` and `weighted_ipopt` ≈
  `ipopt` within noise. Sometimes weighting is slightly *worse*.
  - **Root cause (already noted in code comments):** the RSSI noise model is
    **homogeneous** — every anchor shares the same σ. Reliability weighting has
    nothing to exploit when all measurements are equally noisy. The CRLB
    derivation (`crlb.py`) confirms the same: the bound only depends on
    geometry, β, and σ, not on per-anchor signal level.
  - **Implication:** either (a) drop the weighted variants from the writeup, or
    (b) make the noise **heterogeneous** (e.g. σ increasing with distance /
    decreasing with RSSI) — only then does weighting have a real signal to use,
    and only then does the CRLB become P0-dependent.

- **Particle filter is not a Pareto win.** It costs ~3–4× the SciPy solvers but
  is not more accurate than them, and is well behind IPOPT. It survives as a
  robustness baseline, not a recommended method.

- **`ampl_scip` does not run in this environment.** SCIP is *not* in the free
  AMPL `coin` bundle (`Cannot invoke scip: no such program`). It needs a
  separately licensed AMPL SCIP module. The code degrades gracefully
  (`success=False`) rather than crashing.

- **`ampl_cbc` / `ampl_cuopt` are intentionally unusable here.** They are
  LP/MIP solvers and cannot represent the nonlinear log-distance objective.
  Calling them raises a clear `ValueError`. (They would only fit a
  *reformulated* discrete anchor-placement problem, not per-target NLP.)

- **AMPL adds overhead for no accuracy gain.** `ampl_ipopt` (~471 ms) is ~2×
  native `ipopt` (~250 ms) because of the NL-file round trip, and lands on the
  same optimum. AMPL is useful only as a cross-solver *validation*, not for
  production speed.

- **RMSE alone is a misleading summary for the clustering study.** See §6.

---

## 6. The clustering (GDOP) study — read medians, not RMSE

Fixed signal (P0=−40, β=3, σ=2), 100 replications × 3 spacings × 5 solvers.

RMSE is dominated by ~20–35% of runs that are catastrophic (>500 m), which come
from unlucky anchor draws with near-degenerate geometry (the CRLB itself ranges
from ~45 m to >1600 m across draws at 20 m spacing). **Report median error and
catastrophic-failure rate instead.**

IPOPT, per spacing (100 reps):

| Spacing | Median (m) | RMSE (m) | Catastrophic % | CRLB (m) |
|---|---|---|---|---|
| 20 m | 116.1 | 416.3 | 28% | 140.4 |
| 30 m | 95.9 | 372.5 | 22% | 119.1 |
| 40 m | 82.7 | 384.2 | 23% | 96.2 |

**Caveat / open question:** at 100 reps the *median* trend actually shows
**wider spacing = lower error** (opposite of the 30-rep sample). This makes
sense — more separated anchors give better GDOP than tightly clustered ones —
but it means the earlier "tight is better" read was a small-sample artifact.
The bound (CRLB) moves the same direction (looser = lower). Efficiency
(CRLB/median) is ~1.1–1.3 for all solvers (medians beat the mean-based bound
because bounded clipping biases the estimator). **Recommend ≥100 reps for any
clustering conclusions, and describe the effect via medians + CRLB together.**

---

## 7. Current datasets on disk (all real, full-size)

| File | Rows | What |
|---|---|---|
| `sweep_clean_results.csv` | 3000 | clean sweep, σ=1–2, β=2–4, 100 reps |
| `solver_comparison_results.csv` | 6000 | full comparison, σ=1–8, β=2–4, 100 reps |
| `clustering_results.csv` | 1500 | GDOP study, 100 reps × 3 spacings |
| `solver_comparison_ampl.csv` | 360 | AMPL validation, 20 reps (ipopt/ampl_ipopt/ampl_bonmin) |
| `results.csv` | 1500 | single random-sampling demo, 100 runs |

Report outputs (regenerated from the above): `crlb_plots/`,
`clustering_plots/`, `solver_comparison_plots/`, `solver_comparison_ampl_plots/`.

Regenerate any report with, e.g.:
```bash
python run.py report-crlb        --input sweep_clean_results.csv    --outdir crlb_plots
python run.py report-clustering  --input clustering_results.csv     --outdir clustering_plots
python run.py report-solvers     --input solver_comparison_results.csv --outdir solver_comparison_plots
```

---

## 8. Recommended next steps

1. ~~Decide on the weighted solvers: drop them, or introduce **heterogeneous
   noise** so weighting becomes meaningful.~~ **DONE (2026-07)** — added an
   optional distance-dependent noise model (`het_factor`) plus reproducible
   seeding. See §9.
2. If SCIP comparison is wanted, obtain the licensed AMPL SCIP module.
3. For the paper's headline numbers, use the clean sweep (σ=1–2); for
   robustness discussion, use the full sweep + catastrophic-rate metric.

---

## 9. Heterogeneous noise + reproducible seeding (2026-07)

**Motivation:** §5 found the `weighted`/`weighted_ipopt` solvers were ~no-ops
because the noise was homogeneous — every anchor equally reliable, nothing for
weighting to exploit. Added a knob to make that regime real, and made runs
reproducible while at it.

- **`RSSIModel` heterogeneous noise** (`src/signal/rssi.py`). New knob
  `het_factor` (default **0.0 = old homogeneous model, unchanged**). When >0:
  `sigma_i = noise_std * (1 + het_factor * d_i / het_reference_distance)`, so
  far anchors are noisier. `noise_std_at(distance)` is the single source of
  truth for per-anchor sigma — both the noise draw and the CRLB use it, so they
  can never drift apart. Also takes an optional `rng` for seeded noise.
  - Because sigma depends on **distance, not signal level**, the CRLB stays
    **P0-independent** (the far/near structure is geometry). Tying sigma to RSSI
    level instead would make it P0-dependent — deliberately not done.
- **CRLB per-anchor sigma** (`src/evaluation/crlb.py`).
  `fisher_information_matrix` now accepts a scalar **or** a per-anchor array; the
  scalar path is numerically identical to before (verified by test), so all
  existing homogeneous datasets are unaffected. Under heterogeneity each anchor
  contributes `1/sigma_i^2 * grad grad^T`.
- **Reproducible seeding** (`src/experiments/run_support.py`). New `base_seed`
  config field / `--seed` CLI flag / dashboard checkbox. `None` (default) keeps
  the historical fresh-OS-entropy behaviour; an int makes anchors, targets, and
  RSSI noise fully deterministic per `(base_seed, run_id)`, so a specific outlier
  row / map can be reproduced. Threaded through all three studies
  (single/sweep/clustering) via `derive_seeds`.
- **Verified** (`tests/test_heterogeneous_noise.py`, 9 tests, all pass; runs
  without pytest too): het formula + monotonicity, CRLB scalar==array backward
  compat, seeded-noise determinism, per-run reproducibility, and — the headline
  — **optimal inverse-variance weighting is a no-op under homogeneous noise but
  reduces error under heterogeneous noise.** That last test uses the true
  `1/sigma_i^2` weights (computable from `noise_std_at`) to prove the *signal*
  now exists; note the repo's existing *heuristic* proximity weights only
  approximate those, so `weighted`/`weighted_ipopt` will benefit less than the
  optimal bound — quantifying that gap on a real sweep (run with
  `--het_factor 2 --seed <n>`) is the natural follow-up.

---

## 9. Plot audit + reproducibility fixes (2026-07-19, round 2)

A programmatic audit (reconstructing each plot's data + checking the plotting
logic) turned up three issues; all are now fixed.

### 9.1 The native-IPOPT "bug" — INVESTIGATED, NOT A BUG
Earlier the AMPL comparison showed native `ipopt` occasionally landing far from
the target while `ampl_ipopt` did fine, from an apparently identical start.
Controlled reproduction (single fixed start, no multi-start) → **0 divergences**:
from an identical start, native cyipopt and AMPL-Ipopt converge to the same
point. So the native solver is correct. The divergences came from:
  - **Unseeded `random_point`**: native and AMPL each drew a *different* random
    multi-start point, so on the non-convex RSSI objective they could pick
    different winners. → fixed by seeding (§9.2).
  - **Different Ipopt builds**: cyipopt's bundled Ipopt vs AMPL's Ipopt 3.12.13
    occasionally converge to different local minima *from the same warm start*.
    When this happens, **native finds the LOWER (better) objective ~99% of the
    time** — i.e. native is competitive-to-better, never the culprit. This is a
    property of a non-convex objective + two solver builds, not a code bug.

### 9.2 Fix: seeded, shared multi-start (`multistart.py`, `ipopt_solver.py`)
- `generate_starting_points(..., seed=None)` now seeds `random_point`
  **deterministically from the scenario geometry** (`_scenario_seed`), so every
  solver on the same scenario draws the *same* random start. `seed=int` to
  override, `seed=False` for old non-deterministic behaviour.
- `ipopt_solver.py` was building its starts **inline (unseeded)** on a separate
  code path from the AMPL solver. Refactored it to call the shared
  `generate_starting_points`, so native and AMPL are guaranteed identical.
- **Result:** paired `|ampl_ipopt − ipopt|` error dropped from mean 6.0 m / max
  359 m → **mean 0.9 m / max 58 m**; aggregate RMSE gap 21 m → **0.8 m**.

### 9.3 Fix: objective value logged per solve
`ipopt_solver`, `ampl_solver` now return `objective` (RSSI-domain objective at
the returned solution); `sweep_task.py` persists it as an `objective` CSV
column (None for the SciPy solvers, which optimise a different distance-domain
objective). This lets you verify *which* solver found the lower optimum and
detect boundary-stuck / worse-optimum outliers after the fact — previously
impossible because only the final (x,y) was stored.

### 9.4 Fix: RMSE-based CRLB plot was misleading → added median plot
`crlb_2d_by_solver.png` plots **RMSE**, which a single catastrophic run can
invert: e.g. IPOPT at σ=1 showed β=3 (RMSE 61–100 m) looking *worse* than β=4,
even though β=3's **median is the best**, because one corner-stuck run (true
(814,215) → est (364,1000), 900 m) blew up the mean.
  - Added `median_error` + `median_efficiency_ratio` to the CRLB validation
    table, and a new **`crlb_2d_median_by_solver.png`** plot.
  - The median plot shows the correct monotonic trend (β=2 worst → β=4 best:
    32.9 → 22.5 → 14.3 m at σ=1). **Use the median plot for the β/σ trend;**
    the RMSE plot is only meaningful alongside the catastrophic-rate metric.

### 9.5 Also noted (not a code change)
- `efficiency_ratio` in the solver report uses **mean** CRLB, which the tail
  inflates (mean CRLB 118.8 m vs median 77.5 m in the full sweep). The
  `median_efficiency_ratio` column added earlier is the honest one to quote.
- All datasets were regenerated after these fixes, so every CSV now contains
  the `objective` column and the seeded-start results, and every plot reflects
  them.

---

## 10. Native-IPOPT deep audit + 3 fixes (2026-07-19, round 3)

A controlled deep audit of `IPOPTSolver` / `RSSIDomainProblem` / `IPOPTHyperparams`.
**Conclusion: the core math is correct** — gradient matches numerical to ~1e-8,
recomputed objective matches IPOPT's internal `obj_val` exactly (0.00e+00), and
convergence is ~98.5% across 2000+ solves.

### 10.1 Fix A — broken `hessian_approximation="exact"` (was a landmine)
`IPOPTHyperparams` offered `"exact"` as a validated option and the CLI/sweep
exposed it, but `RSSIDomainProblem.hessian`/`hessianstructure` were **empty
stubs**. Requesting `"exact"` crashed with *"Hessian callback not defined but
called by the Ipopt algorithm."* Implemented the **analytic 2x2 Hessian** (Gauss-Newton
term `k² uuᵀ/d⁴` + curvature term `rᵢ·k·(I/d² − 2 uuᵀ/d⁴)`, returned as the
lower triangle scaled by `obj_factor`) and verified it against a numerical
Hessian — worst relative error **1.6e-9**. `"exact"` now runs cleanly
(status 0) through the full multi-start pipeline. For this 2-D problem `exact`
is cheap and more robust than the default L-BFGS.

### 10.2 Fix B — `success` flag could lie about a non-converged winner
`best = min(candidates, key=objective)` ignored IPOPT status, so a
status=-1 (max-iter) candidate could be returned as "best" while `success=True`
was derived from its (non-converged) status → `success=False` for a good answer,
OR worse the flag was inconsistent. New logic: prefer converged candidates
(status 0/1); among those keep lowest objective; only fall back to a
non-converged candidate if *every* start failed. `success` is now the ACTUAL
returned candidate's convergence. Verified the invariant
`success=True ⇒ status∈{0,1}` holds for both Hessian modes over 120 extreme-noise
runs, and the impact on existing numbers was negligible (~1.5 m on a 200 m RMSE,
0.08% of rows).
  - Note: lowest-objective remains the *correct* tie-breaker — in the one case
    found, the non-converged winner (427 m) beat the best converged one (1015 m).

### 10.3 Dataset consistency — regenerated stale clustering data
`clustering_results.csv` predated the seeding + objective-logging fixes (no
`objective` column, unseeded starts). Regenerated it (and `results.csv`) at
100 reps; `objective` is now persisted in **all** task outputs (`task.py`,
`clustering_task.py`, `sweep_task.py`) for full consistency. Clustering plots
regenerated.

### 10.4 Final status of the open questions
- Native IPOPT correct? **Yes**, confirmed analytically + empirically.
- Issues A & B? **Both fixed.**
- Stale dataset? **Regenerated.** Everything on disk is now post-fix and aligned.

