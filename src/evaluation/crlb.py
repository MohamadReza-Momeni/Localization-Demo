"""
src/evaluation/crlb.py

Cramer-Rao Lower Bound (CRLB) for 2D RSSI-based localization under the
log-distance path-loss model used throughout this project (rssi.py):

    RSSI_i = P0 - 10*n*log10(d_i(theta)/d0) + w_i,   w_i ~ N(0, sigma^2) iid

where theta = (x, y) is the target position, d_i(theta) = ||anchor_i - theta||,
n is the path-loss exponent (beta/PLE), and every anchor shares the same
noise_std sigma (see rssi.py's current homogeneous noise model).

--- Derivation ---
Since w_i is additive Gaussian with theta-independent variance, the Fisher
Information Matrix for theta is:

    FIM_jk = (1/sigma^2) * sum_i  (d mu_i/d theta_j)(d mu_i/d theta_k)

where mu_i(theta) = P0 - 10*n*log10(d_i(theta)/d0) is the noiseless mean.
Differentiating:

    d mu_i/d theta = -k * (theta - anchor_i) / d_i(theta)^2,   k = 10*n/ln(10)

(`k` here is exactly the `log_factor` constant already used in
ipopt_solver.py's RSSIDomainProblem.gradient — this cross-check is a good
sanity test if you re-derive it.)

The CRLB is the inverse of the FIM: Cov(theta_hat) >= FIM^-1 for any
unbiased estimator theta_hat. We report the scalar bound on RMSE as
sqrt(trace(FIM^-1)), i.e. the lower bound on sqrt(E[||theta_hat - theta||^2]).

--- Important consequence worth flagging for your report ---
P0 does not appear in d mu_i/d theta (it's an additive constant term), so
under this model the CRLB is INDEPENDENT of P0 — it only depends on the
anchor geometry, the true position, beta (ple), and sigma. This is a real,
model-driven result, not an omission: it only stops being true if per-anchor
noise becomes signal-LEVEL-dependent (i.e. sigma tied to the measured RSSI,
hence to P0). If you sweep P0 expecting the CRLB curve to move and it doesn't,
that's this, not a bug.

Note on heterogeneous noise (rssi.py's het_factor>0): that model makes sigma
grow with DISTANCE, not with signal level, so `sigma` becomes a per-anchor
array here but the CRLB stays P0-independent — distance doesn't depend on P0.
`fisher_information_matrix` accepts that per-anchor sigma array and weights each
anchor by its own 1/sigma_i^2. This is also the regime where the weighted
solvers finally have a real reliability signal to exploit (PROJECT_NOTES.md
sec.5), because measurements are no longer equally noisy.

--- Degenerate geometry ---
With too few anchors, or anchors that are collinear/poorly distributed
relative to the target (bad GDOP), the FIM can be singular or near-singular.
We detect this via the condition number and return np.inf rather than a
misleadingly huge-but-finite number.
"""
import numpy as np


def _log_factor(ple: float) -> float:
    return 10.0 * ple / np.log(10.0)


def fisher_information_matrix(anchors, true_position, ple, sigma) -> np.ndarray:
    """2x2 FIM for position theta = true_position, given anchor geometry,
    path-loss exponent (ple/beta), and noise std (sigma).

    `sigma` may be either:
      - a scalar: the HOMOGENEOUS model, every anchor shares this noise std.
        FIM = (1/sigma^2) * sum_i grad_i grad_i^T.
      - a per-anchor array (len == len(anchors)): the HETEROGENEOUS model,
        each anchor i contributes (1/sigma_i^2) grad_i grad_i^T. This is the
        correct FIM when noise varies per anchor (rssi.py's het_factor>0),
        because the Fisher information of independent Gaussians simply adds,
        each weighted by its own inverse variance.

    The scalar path is numerically identical to the previous scalar-only
    implementation, so existing (homogeneous) results are unchanged.
    """
    anchors = np.asarray(anchors, dtype=float)
    true_position = np.asarray(true_position, dtype=float)
    k = _log_factor(ple)

    sigma_arr = np.atleast_1d(np.asarray(sigma, dtype=float))
    if sigma_arr.size == 1:
        sigma_per_anchor = np.full(len(anchors), float(sigma_arr[0]))
    elif sigma_arr.size == len(anchors):
        sigma_per_anchor = sigma_arr
    else:
        raise ValueError(
            f"sigma must be a scalar or have one entry per anchor "
            f"(got {sigma_arr.size} for {len(anchors)} anchors)"
        )

    fim = np.zeros((2, 2))
    for anchor, sig in zip(anchors, sigma_per_anchor):
        diff = true_position - anchor
        d2 = float(np.dot(diff, diff))
        d2 = max(d2, 1e-9)  # avoid blow-up if a target sits exactly on an anchor
        grad = -k * diff / d2  # d(mu_i)/d(theta), shape (2,)
        sigma_sq = max(sig ** 2, 1e-12)
        fim += np.outer(grad, grad) / sigma_sq

    return fim


def crlb_covariance(anchors, true_position, ple, sigma, cond_threshold=1e12) -> np.ndarray:
    """Inverse FIM (the CRLB covariance matrix). Returns a matrix of np.inf
    if the geometry makes the FIM singular/ill-conditioned (e.g. <3 anchors,
    or anchors collinear with the target — bad GDOP)."""
    fim = fisher_information_matrix(anchors, true_position, ple, sigma)

    if np.linalg.cond(fim) > cond_threshold:
        return np.full((2, 2), np.inf)

    return np.linalg.inv(fim)


def crlb_rmse(anchors, true_position, ple, sigma) -> float:
    """Scalar CRLB on position RMSE: sqrt(trace(FIM^-1)). This is the number
    to compare directly against empirical sqrt(mean squared error) from
    Monte Carlo runs at the same (ple, sigma) — never lower than this for
    any unbiased estimator, in expectation."""
    cov = crlb_covariance(anchors, true_position, ple, sigma)
    trace = np.trace(cov)
    if not np.isfinite(trace):
        return float("inf")
    return float(np.sqrt(trace))


def crlb_per_axis(anchors, true_position, ple, sigma) -> tuple[float, float]:
    """(sigma_x_bound, sigma_y_bound): CRLB standard deviation bound on each
    axis separately, i.e. sqrt of the diagonal of the CRLB covariance matrix.
    Useful for the "matrix -> specify bounds" style plot (per-axis error
    ellipse) rather than just the combined scalar RMSE bound."""
    cov = crlb_covariance(anchors, true_position, ple, sigma)
    if not np.all(np.isfinite(cov)):
        return float("inf"), float("inf")
    return float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
