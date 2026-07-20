"""
tests/test_ipopt_math.py

Numerical (finite-difference) validation of the analytic derivatives in
src/localization/ipopt_solver.py's RSSIDomainProblem:

  * objective(theta)  -- the RSSI-domain weighted sum-of-squares
  * gradient(theta)   -- analytic 1st derivative  (vs central difference)
  * hessian(theta)    -- analytic exact 2x2 Hessian (vs central difference
                         of the gradient), returned as the lower triangle
                         scaled by obj_factor, in hessianstructure() order

Why this matters (PROJECT_NOTES.md sec.3 / sec.10.1): the analytic gradient is
what IPOPT differentiates, and hessian_approximation="exact" feeds hessian()
directly to IPOPT. If either drifts from the coded objective, IPOPT silently
optimises the wrong function. These tests reconstruct the ground truth purely
from objective() via finite differences, so they can't share a bug with the
analytic code they check.

Runnable two ways:
    pytest tests/                      # if pytest is installed
    python tests/test_ipopt_math.py    # plain-python fallback (no pytest dep)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.localization.ipopt_solver import RSSIDomainProblem


# ---------------------------------------------------------------------------
# Finite-difference reference implementations (depend ONLY on problem.objective)
# ---------------------------------------------------------------------------
def fd_gradient(problem, theta, h=1e-6):
    """Central-difference gradient of problem.objective at theta."""
    theta = np.asarray(theta, dtype=float)
    grad = np.zeros(2)
    for i in range(2):
        step = np.zeros(2)
        step[i] = h
        f_plus = problem.objective(theta + step)
        f_minus = problem.objective(theta - step)
        grad[i] = (f_plus - f_minus) / (2.0 * h)
    return grad


def fd_hessian(problem, theta, h=1e-5):
    """Central-difference Hessian, built from the ANALYTIC gradient so we test
    hessian() against a derivative of gradient() (both must be self-consistent
    with objective(), which fd_gradient already pins down)."""
    theta = np.asarray(theta, dtype=float)
    H = np.zeros((2, 2))
    for j in range(2):
        step = np.zeros(2)
        step[j] = h
        g_plus = problem.gradient(theta + step)
        g_minus = problem.gradient(theta - step)
        H[:, j] = (g_plus - g_minus) / (2.0 * h)
    # symmetrise (analytic Hessian is exactly symmetric; FD has tiny asymmetry)
    return 0.5 * (H + H.T)


def analytic_hessian_full(problem, theta, obj_factor=1.0):
    """Reassemble the full 2x2 from the lower-triangle vector hessian() returns,
    in the order declared by hessianstructure(): (0,0),(1,0),(1,1)."""
    rows, cols = problem.hessianstructure()
    vals = problem.hessian(np.asarray(theta, dtype=float),
                           lagrange=np.array([]), obj_factor=obj_factor)
    H = np.zeros((2, 2))
    for r, c, v in zip(rows, cols, vals):
        H[r, c] = v
        H[c, r] = v  # mirror lower -> upper
    return H


# ---------------------------------------------------------------------------
# Scenario fixtures (deterministic; a spread of geometries, PLE, and weights)
# ---------------------------------------------------------------------------
def _make_problem(seed, weighted=False, n_anchors=6):
    rng = np.random.default_rng(seed)
    anchors = rng.uniform(0, 1000, size=(n_anchors, 2))
    true_pos = rng.uniform(100, 900, size=2)
    p0 = rng.uniform(-50, 0)
    ple = rng.uniform(2.0, 4.0)

    # Build noiseless-ish distances from a true position, then add mild noise so
    # residuals are non-zero (the curvature term r_i*k*(...) in the Hessian only
    # exercises when residuals != 0 -- a zero-residual fit would hide bugs there).
    true_dists = np.linalg.norm(anchors - true_pos, axis=1)
    noise = rng.normal(0, 1.5, size=n_anchors)
    # Perturb distances multiplicatively to keep them positive.
    distances = true_dists * np.exp(noise / (10.0 * ple))

    weights = None
    if weighted:
        weights = np.clip(rng.uniform(0.2, 5.0, size=n_anchors), 0.2, 5.0)

    problem = RSSIDomainProblem(anchors, distances, ref_power=p0, ple=ple, weights=weights)
    # Evaluation points: the true position, an off-target point, and the centroid.
    eval_points = [
        true_pos,
        true_pos + rng.uniform(-150, 150, size=2),
        np.mean(anchors, axis=0),
    ]
    return problem, [np.clip(p, 1.0, 999.0) for p in eval_points]


SCENARIOS = []
for _seed in range(8):
    SCENARIOS.append((_seed, False))
    SCENARIOS.append((_seed, True))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_gradient_matches_finite_difference():
    """Analytic gradient == central-difference gradient to ~1e-6 relative."""
    worst = 0.0
    for seed, weighted in SCENARIOS:
        problem, points = _make_problem(seed, weighted)
        for theta in points:
            g_analytic = problem.gradient(theta)
            g_numeric = fd_gradient(problem, theta)
            denom = max(np.linalg.norm(g_numeric), 1.0)
            rel = np.linalg.norm(g_analytic - g_numeric) / denom
            worst = max(worst, rel)
            assert rel < 1e-5, (
                f"gradient mismatch seed={seed} weighted={weighted} theta={theta}: "
                f"analytic={g_analytic} numeric={g_numeric} rel={rel:.2e}"
            )
    print(f"  [gradient] worst relative error = {worst:.2e}")


def test_hessian_matches_finite_difference():
    """Analytic exact Hessian == central-difference of the gradient (~1e-5)."""
    worst = 0.0
    for seed, weighted in SCENARIOS:
        problem, points = _make_problem(seed, weighted)
        for theta in points:
            H_analytic = analytic_hessian_full(problem, theta, obj_factor=1.0)
            H_numeric = fd_hessian(problem, theta)
            denom = max(np.linalg.norm(H_numeric), 1.0)
            rel = np.linalg.norm(H_analytic - H_numeric) / denom
            worst = max(worst, rel)
            assert rel < 1e-4, (
                f"hessian mismatch seed={seed} weighted={weighted} theta={theta}: "
                f"analytic=\n{H_analytic}\nnumeric=\n{H_numeric}\nrel={rel:.2e}"
            )
    print(f"  [hessian] worst relative error = {worst:.2e}")


def test_hessian_is_symmetric():
    """hessianstructure() declares the lower triangle; the reassembled matrix
    must be symmetric (H[0,1] == H[1,0])."""
    for seed, weighted in SCENARIOS:
        problem, points = _make_problem(seed, weighted)
        for theta in points:
            H = analytic_hessian_full(problem, theta)
            assert np.isclose(H[0, 1], H[1, 0]), f"asymmetric Hessian at {theta}"


def test_hessian_obj_factor_scales_linearly():
    """cyipopt multiplies the objective Hessian by obj_factor; hessian() must
    honour that (the returned values scale linearly in obj_factor)."""
    problem, points = _make_problem(0, weighted=True)
    theta = points[1]
    h1 = problem.hessian(theta, lagrange=np.array([]), obj_factor=1.0)
    h3 = problem.hessian(theta, lagrange=np.array([]), obj_factor=3.0)
    assert np.allclose(h3, 3.0 * h1), "obj_factor does not scale the Hessian linearly"


def test_objective_matches_manual_definition():
    """objective() equals the textbook weighted SSE of RSSI residuals, computed
    independently here so the objective the derivatives are checked against is
    itself pinned to the documented model."""
    for seed, weighted in SCENARIOS:
        problem, points = _make_problem(seed, weighted)
        for theta in points:
            theta = np.asarray(theta, dtype=float)
            dists = np.linalg.norm(problem.anchors - theta, axis=1)
            rssi_pred = problem.ref_power - 10.0 * problem.ple * np.log10(dists)
            residuals = problem.rssi_meas - rssi_pred
            manual = np.sum(problem.weights * residuals ** 2)
            got = problem.objective(theta)
            # 1e-9 epsilons inside objective() make this approximate, not exact.
            assert np.isclose(got, manual, rtol=1e-6, atol=1e-6), (
                f"objective mismatch seed={seed}: got={got} manual={manual}"
            )


def test_objective_minimised_near_true_when_noiseless():
    """Sanity: with (near) zero noise the objective at the true position is
    tiny and lower than at a displaced point -- confirms the residual sign /
    model orientation is right (a flipped sign would put the max at the truth)."""
    rng = np.random.default_rng(123)
    anchors = rng.uniform(0, 1000, size=(6, 2))
    true_pos = np.array([500.0, 500.0])
    p0, ple = -40.0, 3.0
    dists = np.linalg.norm(anchors - true_pos, axis=1)
    problem = RSSIDomainProblem(anchors, dists, ref_power=p0, ple=ple)
    f_true = problem.objective(true_pos)
    f_off = problem.objective(true_pos + np.array([120.0, -80.0]))
    assert f_true < 1e-3, f"noiseless objective at truth should be ~0, got {f_true}"
    assert f_off > f_true, "objective should increase away from the true position"


if __name__ == "__main__":
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in _tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}\n      {e}")
    print(f"\n{len(_tests) - failures}/{len(_tests)} passed")
    sys.exit(1 if failures else 0)
