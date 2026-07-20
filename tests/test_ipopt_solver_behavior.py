"""
tests/test_ipopt_solver_behavior.py

End-to-end checks that the two audit fixes from PROJECT_NOTES.md sec.10 are
actually wired through the live cyipopt solve path (not just present in source):

  Fix A (sec.10.1) -- hessian_approximation="exact" runs cleanly instead of
    crashing with "Hessian callback not defined". Confirms hessianstructure()/
    hessian() are hooked up and IPOPT accepts them, and that "exact" reaches the
    same optimum as "limited-memory".

  Fix B (sec.10.2) -- the returned `success` flag reflects the ACTUAL returned
    candidate's convergence: success=True  =>  info status in {0, 1}. A
    non-converged winner must not be reported as success=True.

Also checks the `objective` column plumbing: the solver returns a finite
`objective` equal to problem.objective(returned solution), and SolverRegistry
propagates both `success` and `objective` into a result row.

These require cyipopt; if it's missing the whole module is skipped (plain-python
runner treats that as a skip, not a failure).

Runnable two ways:
    pytest tests/
    python tests/test_ipopt_solver_behavior.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.localization.ipopt_solver import IPOPTSolver, RSSIDomainProblem
from src.localization.ipopt_params import IPOPTHyperparams
from src.experiments.strategies import SolverRegistry
from src.experiments.context import RunContext

try:
    import cyipopt  # noqa: F401
    HAVE_CYIPOPT = True
except ImportError:
    HAVE_CYIPOPT = False


class SkipTest(Exception):
    """Raised to signal a skip in the plain-python runner."""


def _require_cyipopt():
    if not HAVE_CYIPOPT:
        raise SkipTest("cyipopt not installed")


# ---------------------------------------------------------------------------
# Deterministic scenario
# ---------------------------------------------------------------------------
def _scenario(seed=7, n_anchors=6):
    rng = np.random.default_rng(seed)
    anchors = rng.uniform(0, 1000, size=(n_anchors, 2))
    true_pos = rng.uniform(150, 850, size=2)
    p0, ple = -40.0, 3.0
    true_dists = np.linalg.norm(anchors - true_pos, axis=1)
    noise = rng.normal(0, 1.0, size=n_anchors)
    distances = true_dists * np.exp(noise / (10.0 * ple))
    return anchors, distances, true_pos, p0, ple


# ---------------------------------------------------------------------------
# Fix A -- exact Hessian
# ---------------------------------------------------------------------------
def test_fix_a_exact_hessian_runs():
    """hessian_approximation='exact' completes without the 'Hessian callback
    not defined' crash and converges (status in {0,1})."""
    _require_cyipopt()
    anchors, distances, true_pos, p0, ple = _scenario()
    solver = IPOPTSolver()
    hp = IPOPTHyperparams(hessian_approximation="exact",
                          starting_points=("warm_start",), multi_start=False)
    sol = solver.solve(anchors, distances, ref_power=p0, ple=ple,
                       x0=np.mean(anchors, axis=0), hyperparams=hp)
    assert sol["info"]["status"] in (0, 1), f"exact-Hessian solve did not converge: {sol['info']['status']}"
    assert np.all(np.isfinite(sol["solution"]))
    print(f"  [Fix A] exact-Hessian solve status={sol['info']['status']} "
          f"solution={sol['solution']}")


def test_fix_a_exact_matches_limited_memory():
    """'exact' and 'limited-memory' reach essentially the same optimum on this
    smooth problem -- both must find the same minimiser to within a few metres."""
    _require_cyipopt()
    anchors, distances, true_pos, p0, ple = _scenario()
    solver = IPOPTSolver()
    x0 = np.mean(anchors, axis=0)
    common = dict(starting_points=("warm_start",), multi_start=False)
    sol_exact = solver.solve(anchors, distances, ref_power=p0, ple=ple, x0=x0,
                             hyperparams=IPOPTHyperparams(hessian_approximation="exact", **common))
    sol_lbfgs = solver.solve(anchors, distances, ref_power=p0, ple=ple, x0=x0,
                             hyperparams=IPOPTHyperparams(hessian_approximation="limited-memory", **common))
    diff = np.linalg.norm(sol_exact["solution"] - sol_lbfgs["solution"])
    assert diff < 5.0, f"exact vs limited-memory diverged by {diff:.3f} m"
    # objectives should agree even more tightly than positions
    assert abs(sol_exact["objective"] - sol_lbfgs["objective"]) < 1e-3
    print(f"  [Fix A] exact vs limited-memory: |d_position|={diff:.3e} m")


# ---------------------------------------------------------------------------
# Fix B -- honest success flag
# ---------------------------------------------------------------------------
def test_fix_b_success_implies_converged_status():
    """Invariant success=True => status in {0,1}, across many scenarios and both
    Hessian modes. This is the core of Fix B: the flag must describe the actually
    returned candidate, never a non-converged winner."""
    _require_cyipopt()
    solver = IPOPTSolver()
    checked = 0
    for seed in range(12):
        anchors, distances, true_pos, p0, ple = _scenario(seed=seed)
        for hess in ("limited-memory", "exact"):
            hp = IPOPTHyperparams(hessian_approximation=hess)
            sol = solver.solve(anchors, distances, ref_power=p0, ple=ple,
                               x0=np.mean(anchors, axis=0), hyperparams=hp)
            status = sol["info"]["status"]
            if sol["success"]:
                assert status in (0, 1), (
                    f"success=True but status={status} (seed={seed}, hess={hess})"
                )
            checked += 1
    print(f"  [Fix B] success=>converged invariant held over {checked} solves")


def test_fix_b_returned_objective_matches_solution():
    """The reported `objective` equals problem.objective(returned solution) --
    i.e. the objective column describes the point actually returned (the lowest-
    objective candidate), not some other candidate."""
    _require_cyipopt()
    anchors, distances, true_pos, p0, ple = _scenario(seed=3)
    solver = IPOPTSolver()
    sol = solver.solve(anchors, distances, ref_power=p0, ple=ple,
                       x0=np.mean(anchors, axis=0), hyperparams=IPOPTHyperparams())
    problem = RSSIDomainProblem(anchors, distances, ref_power=p0, ple=ple)
    recomputed = problem.objective(sol["solution"])
    assert np.isfinite(sol["objective"])
    assert np.isclose(sol["objective"], recomputed, rtol=1e-9, atol=1e-9), (
        f"reported objective {sol['objective']} != recomputed {recomputed}"
    )
    print(f"  [Fix B] reported objective matches solution: {sol['objective']:.6f}")


def test_fix_b_winner_is_lowest_objective_among_converged():
    """When >=1 candidate converged, the returned solution is the lowest-objective
    converged candidate (the documented tie-break in ipopt_solver.py)."""
    _require_cyipopt()
    anchors, distances, true_pos, p0, ple = _scenario(seed=5)
    solver = IPOPTSolver()
    sol = solver.solve(anchors, distances, ref_power=p0, ple=ple,
                       x0=np.mean(anchors, axis=0), hyperparams=IPOPTHyperparams())
    cands = sol["multi_start_candidates"]
    converged = [c for c in cands if c["status"] in (0, 1)]
    pool = converged if converged else cands
    best_obj = min(c["objective"] for c in pool)
    assert np.isclose(sol["objective"], best_obj), (
        f"returned objective {sol['objective']} is not the min over the "
        f"correct pool ({best_obj})"
    )


# ---------------------------------------------------------------------------
# Column plumbing through SolverRegistry
# ---------------------------------------------------------------------------
def test_registry_propagates_success_and_objective():
    """SolverRegistry.execute_solver('ipopt', ...) returns a dict carrying a
    boolean `success`, a finite `objective`, and a `solve_time_sec` -- the three
    things task.py writes into the CSV row for the IPOPT family."""
    _require_cyipopt()
    anchors, distances, true_pos, p0, ple = _scenario(seed=9)
    x_range, y_range = (0, 1000), (0, 1000)
    registry = SolverRegistry(x_range, y_range)
    baseline = registry.execute_solver("vanilla", RunContext(
        anchors, distances, None, p0, ple, x_range, y_range))["solution"]
    ctx = RunContext(anchors, distances, baseline, p0, ple, x_range, y_range)
    sol = registry.execute_solver("ipopt", ctx)
    assert isinstance(sol["success"], (bool, np.bool_)), "success must be boolean"
    assert np.isfinite(sol["objective"]), "objective must be finite"
    assert sol["solve_time_sec"] >= 0.0
    assert "chosen_start" in sol, "IPOPT row must record which start won"
    if sol["success"]:
        assert sol["info"]["status"] in (0, 1)
    print(f"  [plumbing] ipopt row: success={sol['success']} "
          f"objective={sol['objective']:.4f} start={sol['chosen_start']}")


if __name__ == "__main__":
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = skips = 0
    for t in _tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except SkipTest as e:
            skips += 1
            print(f"SKIP  {t.__name__}  ({e})")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}\n      {e}")
    passed = len(_tests) - failures - skips
    print(f"\n{passed}/{len(_tests)} passed, {skips} skipped, {failures} failed")
    sys.exit(1 if failures else 0)
