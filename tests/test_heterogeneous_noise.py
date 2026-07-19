"""
tests/test_heterogeneous_noise.py

Validates the two engineering features added on top of the verified IPOPT math:

  1. Heterogeneous (distance-dependent) noise (src/signal/rssi.py)
     - het_factor=0 is identical to the old homogeneous model.
     - het_factor>0 makes far anchors noisier via a single source of truth
       (noise_std_at), which the CRLB also consumes.
     - Under heterogeneity the WEIGHTED distance-domain solver beats the
       unweighted one on average (the point of the feature).

  2. CRLB per-anchor sigma (src/evaluation/crlb.py)
     - scalar sigma path is numerically identical to a constant per-anchor
       array (backward compatibility with every existing dataset);
     - a per-anchor array is accepted and moves the bound as expected.

  3. Reproducible per-run seeding (run_support.py + task.py)
     - with base_seed set, SimulationTask.execute(run_id) is deterministic;
     - with base_seed=None the historical fresh-entropy behaviour is kept.

Runnable two ways (pytest is NOT installed here):
    pytest tests/
    python tests/test_heterogeneous_noise.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signal.rssi import RSSIModel
from src.evaluation.crlb import fisher_information_matrix, crlb_rmse
from src.experiments.config import ExperimentConfig
from src.experiments.task import SimulationTask
from src.experiments.strategies import SolverRegistry
from src.signal.distance import rssi_to_distance


def _demo_anchors():
    return np.array([[0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0], [1000.0, 1000.0]])


# --- 1. RSSIModel noise model ---
def test_homogeneous_noise_std_is_flat():
    m = RSSIModel(noise_std=2.0, het_factor=0.0)
    for d in (1.0, 50.0, 500.0, 1e4):
        assert m.noise_std_at(d) == 2.0, f"homogeneous sigma changed at d={d}"
    assert np.allclose(m.noise_std_at(np.array([1.0, 100.0, 900.0])), 2.0)


def test_heterogeneous_noise_std_grows_with_distance():
    m = RSSIModel(noise_std=2.0, het_factor=1.0, het_reference_distance=100.0)
    assert m.noise_std_at(0.0) == 2.0
    assert np.isclose(m.noise_std_at(100.0), 4.0)   # 2*(1 + 100/100)
    assert np.isclose(m.noise_std_at(200.0), 6.0)   # 2*(1 + 200/100)
    sig = m.noise_std_at(np.array([0.0, 10.0, 100.0, 1000.0]))
    assert np.all(np.diff(sig) > 0), "sigma must strictly increase with distance"


def test_seeded_noise_is_reproducible():
    d = 123.0
    a = RSSIModel(noise_std=2.0, rng=np.random.default_rng(7)).rssi(d)
    b = RSSIModel(noise_std=2.0, rng=np.random.default_rng(7)).rssi(d)
    c = RSSIModel(noise_std=2.0, rng=np.random.default_rng(8)).rssi(d)
    assert a == b, "same seed must reproduce the same noisy RSSI"
    assert a != c, "different seed should (almost surely) differ"


# --- 2. CRLB per-anchor sigma ---
def test_crlb_scalar_matches_constant_array():
    anchors = _demo_anchors()
    target = np.array([400.0, 300.0])
    ple, sigma = 3.0, 2.5
    fim_scalar = fisher_information_matrix(anchors, target, ple, sigma)
    fim_array = fisher_information_matrix(anchors, target, ple, np.full(len(anchors), sigma))
    assert np.array_equal(fim_scalar, fim_array), "scalar vs constant-array FIM drifted"
    assert crlb_rmse(anchors, target, ple, sigma) == \
        crlb_rmse(anchors, target, ple, np.full(len(anchors), sigma)), "CRLB RMSE drifted"


def test_crlb_per_anchor_array_changes_bound():
    anchors = _demo_anchors()
    target = np.array([400.0, 300.0])
    base = crlb_rmse(anchors, target, 3.0, 2.0)
    het = crlb_rmse(anchors, target, 3.0, np.array([2.0, 4.0, 6.0, 8.0]))
    assert het > base, "raising some anchors' sigma should loosen the CRLB"


def test_crlb_rejects_wrong_length_sigma():
    anchors = _demo_anchors()
    target = np.array([400.0, 300.0])
    try:
        fisher_information_matrix(anchors, target, 3.0, np.array([1.0, 2.0]))
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched sigma length")


# --- 3. Weighting helps under heterogeneous noise ---
def _mean_uniform_vs_invvar(het_factor, n_trials=150, base_seed=2024):
    """Return (mean_uniform_error, mean_invvar_error) for the distance-domain
    SciPy solver over many seeded noise realisations, comparing UNIFORM weights
    against optimal INVERSE-VARIANCE weights (1/sigma_i^2, the statistically
    correct reliability weighting the het model now makes computable via
    noise_std_at). Uses only the SciPy solver, so no cyipopt/AMPL needed.

    Feeding the true 1/sigma_i^2 isolates the QUESTION THE FEATURE ANSWERS —
    'is there now a reliability signal for weighting to exploit?' — from the
    separate, weaker question of whether the repo's heuristic weight functions
    happen to approximate it (PROJECT_NOTES.md sec.5)."""
    solver = SolverRegistry(x_range=(0, 1000), y_range=(0, 1000)).scipy_solver
    anchors = np.array([[100.0, 100.0], [950.0, 100.0], [100.0, 950.0], [950.0, 950.0]])
    true = np.array([200.0, 200.0])
    p0, ple = -40.0, 3.0
    anchor_dists = np.linalg.norm(anchors - true, axis=1)

    unif, inv = [], []
    for t in range(n_trials):
        rng = np.random.default_rng([base_seed, t])
        model = RSSIModel(reference_power=p0, path_loss_exponent=ple, noise_std=1.5,
                          het_factor=het_factor, het_reference_distance=100.0, rng=rng)
        rssi = np.array([model.rssi(d) for d in anchor_dists])
        distances = np.array([rssi_to_distance(r, p0, ple) for r in rssi])

        # Optimal inverse-variance weights, from the SAME per-anchor sigma the
        # model used to generate the noise (noise_std_at is the single source).
        sigma_i = model.noise_std_at(anchor_dists)
        w_invvar = (1.0 / sigma_i ** 2)
        w_invvar = w_invvar / np.mean(w_invvar)

        e_unif = solver.solve(anchors, distances, x_range=(0, 1000), y_range=(0, 1000))["solution"]
        e_inv = solver.solve(anchors, distances, weights=w_invvar,
                             x_range=(0, 1000), y_range=(0, 1000))["solution"]
        unif.append(np.linalg.norm(e_unif - true))
        inv.append(np.linalg.norm(e_inv - true))

    return float(np.mean(unif)), float(np.mean(inv))


def test_reliability_weighting_exploitable_only_under_heterogeneous_noise():
    """The engineering point of the het-noise feature.

    Homogeneous noise: every sigma_i is equal, so inverse-variance weights are
    uniform -> the weighted solve is IDENTICAL to the unweighted one. There is,
    by construction, no reliability signal to exploit (matching PROJECT_NOTES
    sec.5's finding that weighting was a no-op).

    Heterogeneous noise: sigma_i varies with range, so inverse-variance weights
    differ and down-weight the genuinely-noisier far anchors, reducing mean
    error. That gain is the capability this feature adds."""
    unif_hom, inv_hom = _mean_uniform_vs_invvar(het_factor=0.0)
    unif_het, inv_het = _mean_uniform_vs_invvar(het_factor=4.0)

    # Homogeneous: inverse-variance weights collapse to uniform -> no change.
    assert np.isclose(unif_hom, inv_hom, rtol=1e-6), (
        f"under homogeneous noise weighting should be a no-op "
        f"(uniform={unif_hom:.4f}, inv-var={inv_hom:.4f})"
    )
    # Heterogeneous: correct reliability weighting reduces mean error.
    assert inv_het < unif_het, (
        f"inverse-variance weighting did NOT help under heterogeneous noise "
        f"(uniform={unif_het:.2f}, inv-var={inv_het:.2f})"
    )


# --- 4. Reproducible per-run seeding ---
def test_simulation_task_reproducible_with_seed():
    """Same base_seed + run_id -> identical rows; different base_seed -> different."""
    cfg = ExperimentConfig(anchor_count=4, target_count=1, base_seed=99,
                           solvers=("vanilla",))
    r1 = SimulationTask(cfg).execute(0)
    r2 = SimulationTask(cfg).execute(0)
    assert r1[0]["true_x"] == r2[0]["true_x"] and r1[0]["true_y"] == r2[0]["true_y"], \
        "seeded target not reproducible"
    assert r1[0]["anchors"] == r2[0]["anchors"], "seeded anchors not reproducible"
    assert r1[0]["est_x"] == r2[0]["est_x"], "seeded estimate not reproducible"

    cfg2 = ExperimentConfig(anchor_count=4, target_count=1, base_seed=100,
                            solvers=("vanilla",))
    r3 = SimulationTask(cfg2).execute(0)
    assert r1[0]["anchors"] != r3[0]["anchors"], "different seed should give different anchors"


def test_simulation_task_different_run_ids_differ():
    """Distinct run_ids under one seed are independent draws."""
    cfg = ExperimentConfig(anchor_count=4, target_count=1, base_seed=7, solvers=("vanilla",))
    r0 = SimulationTask(cfg).execute(0)
    r1 = SimulationTask(cfg).execute(1)
    assert r0[0]["anchors"] != r1[0]["anchors"], "different run_ids should differ"


# --- dual-mode runner (no pytest dependency) ---
if __name__ == "__main__":
    tests = [
        test_homogeneous_noise_std_is_flat,
        test_heterogeneous_noise_std_grows_with_distance,
        test_seeded_noise_is_reproducible,
        test_crlb_scalar_matches_constant_array,
        test_crlb_per_anchor_array_changes_bound,
        test_crlb_rejects_wrong_length_sigma,
        test_reliability_weighting_exploitable_only_under_heterogeneous_noise,
        test_simulation_task_reproducible_with_seed,
        test_simulation_task_different_run_ids_differ,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {t.__name__}: {exc}")
            failed += 1
    print(f"\n{passed}/{len(tests)} passed, {failed} failed")
    sys.exit(1 if failed else 0)
