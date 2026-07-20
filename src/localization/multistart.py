"""
src/localization/multistart.py

Shared multi-start candidate generation, used by both IPOPTSolver (inline,
see ipopt_solver.py) and AMPLSolver (via this function, see ampl_solver.py).
Factored out so both solve() paths build IDENTICAL starting points from
IDENTICAL inputs — that's what makes a comparison between "ipopt" (native
cyipopt) and "ampl_ipopt"/"ampl_bonmin"/"ampl_scip" (AMPL-routed) isolate
the solver/algorithm rather than picking up incidental differences in where
each one started from.

Three candidate starting points, matching IPOPTHyperparams.starting_points'
valid labels (ipopt_params.py):
  - "warm_start": the caller-supplied x0, or the anchor centroid if no x0
    was given (e.g. this IS the first/only solver call in the pipeline).
  - "anchor_centroid": mean of anchor positions — a geometry-only guess
    that doesn't depend on any previous solver's output.
  - "random_point": uniform random point inside the given x_range/y_range,
    guards against every deterministic start sharing the same bad local
    basin on this non-convex RSSI objective.

Reproducibility / fair cross-solver comparison
----------------------------------------------
The "random_point" start is DETERMINISTIC by default: its seed is derived
from the scenario itself (anchor geometry + map bounds) via `_scenario_seed`.
This matters because "ipopt" (native) and "ampl_ipopt"/"ampl_bonmin" are
supposed to be compared on an identical problem — if each one drew a fresh
unseeded random start, they would silently start from different points and a
non-convex objective could then send them to different local minima, making
the comparison measure the RNG rather than the solver. With a scenario-derived
seed, every solver run on the same scenario draws the SAME random point, so
the comparison is apples-to-apples and fully reproducible across processes.

Pass an explicit integer `seed` to override, or `seed=False` to opt back into
non-deterministic behaviour (e.g. if you deliberately want independent random
restarts across repeated calls on one scenario).
"""
import numpy as np


def _scenario_seed(anchors, x_range, y_range) -> int:
    """Stable 32-bit seed derived from the scenario geometry, so the same
    anchors+bounds always yield the same 'random' start regardless of which
    solver (or process) asks for it. Not cryptographic — just a reproducible
    hash of the float inputs."""
    key = np.concatenate([
        np.asarray(anchors, dtype=float).ravel(),
        np.asarray([x_range[0], x_range[1], y_range[0], y_range[1]], dtype=float),
    ])
    # Hash the raw bytes; mask to 32 bits for numpy's SeedSequence.
    return int(abs(hash(key.tobytes()))) & 0xFFFFFFFF


def generate_starting_points(anchors, x0, x_range, y_range,
                             labels=("warm_start", "anchor_centroid", "random_point"),
                             seed=None):
    """Returns {label: np.array([x, y])} for every label in `labels`.

    Only computes what's asked for in `labels` (so a caller that only wants
    "warm_start" doesn't pay for a wasted random draw), but all three are
    cheap enough that computing the full set is also fine if you'd rather
    always pass the default three and let the solver pick which to run.

    `seed`:
      - None (default): the random_point is seeded reproducibly from the
        scenario geometry (see module docstring), so all solvers agree.
      - int: use exactly this seed for the random_point draw.
      - False: draw a fresh non-deterministic random_point each call.
    """
    anchors = np.asarray(anchors, dtype=float)
    points = {}

    if "warm_start" in labels:
        if x0 is not None:
            points["warm_start"] = np.asarray(x0, dtype=float)
        else:
            # No warm start available (e.g. this is the initial vanilla-equivalent
            # call) — centroid doubles as both the "no info" default and one of
            # the multi-start points, same convention as IPOPTSolver.solve.
            points["warm_start"] = np.mean(anchors, axis=0)

    if "anchor_centroid" in labels:
        points["anchor_centroid"] = np.mean(anchors, axis=0)

    if "random_point" in labels:
        if seed is False:
            rng = np.random.default_rng()
        elif seed is None:
            rng = np.random.default_rng(_scenario_seed(anchors, x_range, y_range))
        else:
            rng = np.random.default_rng(int(seed))
        points["random_point"] = np.array([
            rng.uniform(x_range[0], x_range[1]),
            rng.uniform(y_range[0], y_range[1]),
        ])

    return points