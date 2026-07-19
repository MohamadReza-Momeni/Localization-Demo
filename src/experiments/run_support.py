"""
src/experiments/run_support.py

Small shared helpers used by all three study tasks (task.py / sweep_task.py /
clustering_task.py) so the reproducibility and result-row plumbing lives in ONE
place instead of being copy-pasted three times.

Two concerns:

1. Reproducible per-run seeding (`derive_seeds`)
   -------------------------------------------
   The tasks previously generated anchors/targets with `seed=None` (fresh OS
   entropy) and drew RSSI noise from the GLOBAL `np.random` state, so a run
   could never be reproduced — even re-running the exact same command gave a
   different anchor layout, target, and noise realisation. That defeats
   debugging a specific outlier row and makes the maps non-reproducible.

   `derive_seeds(base_seed, run_id, n)` returns `n` INDEPENDENT numpy
   SeedSequences for one run:
     - base_seed is None  -> seeded from OS entropy (non-reproducible, the old
       default behaviour is preserved unless the user opts in);
     - base_seed is an int -> fully deterministic: run_id R of a study with
       base_seed B always yields the byte-identical anchors/targets/noise,
       while different run_ids stay independent (SeedSequence mixes [B, R]).

   Each returned child is a valid seed for `np.random.default_rng(...)` and for
   the scenario generators' `seed=` argument.

2. Result-row augmentation (`augment_ipopt_ampl_columns`)
   -----------------------------------------------------
   The IPOPT-family / AMPL-family bookkeeping columns were duplicated verbatim
   in task.py and sweep_task.py. Centralised here so a change to what gets
   logged happens once.
"""
import numpy as np


def derive_seeds(base_seed, run_id, n):
    """Return `n` independent SeedSequence children for one run.

    base_seed None -> OS entropy (non-reproducible); int -> deterministic,
    derived from (base_seed, run_id) so runs are reproducible yet distinct.
    """
    if base_seed is None:
        parent = np.random.SeedSequence()
    else:
        parent = np.random.SeedSequence([int(base_seed), int(run_id)])
    return parent.spawn(n)


def augment_ipopt_ampl_columns(row, solution, ipopt_params, include_ipopt_params=True):
    """Add the IPOPT-family / AMPL-family logging columns to `row` in place.

    Mirrors what task.py/sweep_task.py logged before this was factored out:
      - if the solution came from an IPOPT-family solver ("chosen_start" key),
        record which start won and (when include_ipopt_params) the internal
        hyperparameters actually used;
      - if it was AMPL-routed ("solver_name" key), record which underlying
        solver AMPL used plus its chosen start.

    include_ipopt_params=False matches clustering_task.py, which historically
    logged only the chosen start (the hyperparameters are fixed there).
    """
    if "chosen_start" in solution:
        row["ipopt_chosen_start"] = solution["chosen_start"]
        if include_ipopt_params:
            row["ipopt_tol"] = ipopt_params.tol
            row["ipopt_acceptable_tol"] = ipopt_params.acceptable_tol
            row["ipopt_max_iter"] = ipopt_params.max_iter
            row["ipopt_hessian_approximation"] = ipopt_params.hessian_approximation
            row["ipopt_mu_strategy"] = ipopt_params.mu_strategy
            row["ipopt_multi_start"] = ipopt_params.multi_start

    if "solver_name" in solution:
        row["ampl_solver_name"] = solution["solver_name"]
        row["ampl_chosen_start"] = solution.get("chosen_start")

    return row
