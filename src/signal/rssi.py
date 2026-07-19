import numpy as np


class RSSIModel:
    """Forward RSSI model under the log-distance path-loss law.

    Noise model
    -----------
    By default the noise is HOMOGENEOUS: every anchor's measurement shares the
    same std-dev `noise_std` (sigma), regardless of range. This is the model the
    CRLB derivation (crlb.py) and the existing datasets assume, and under it
    reliability weighting has nothing to exploit — see PROJECT_NOTES.md sec.5.

    Set `het_factor > 0` to switch on DISTANCE-DEPENDENT (heterogeneous) noise:

        sigma_i = noise_std * (1 + het_factor * d_i / het_reference_distance)

    i.e. farther anchors are noisier (weaker SNR -> larger RSSI variance), which
    is the physically-motivated regime in which the `weighted`/`weighted_ipopt`
    solvers actually have a reliability signal to exploit. Because sigma_i
    depends only on geometry (distance), not on P0, the CRLB stays P0-independent
    even in the heterogeneous case (see crlb.py).

    `het_factor = 0.0` (the default) reproduces the original homogeneous model
    exactly — `noise_std_at` returns `noise_std` for every anchor — so existing
    studies/CSVs are unaffected unless the knob is turned on explicitly.

    Reproducibility
    ---------------
    Pass an `rng` (a numpy Generator) to draw noise from a caller-controlled,
    seeded stream instead of the global `np.random` state. This is what lets a
    per-run seed make an entire simulation reproducible; if `rng` is None the
    old global-`np.random.normal` behaviour is kept.
    """

    def __init__(
        self,
        reference_power=-40,
        path_loss_exponent=2.2,
        noise_std=2.0,
        reference_distance=1.0,
        het_factor=0.0,
        het_reference_distance=100.0,
        rng=None,
    ):
        if het_factor < 0:
            raise ValueError("het_factor must be >= 0")
        if het_reference_distance <= 0:
            raise ValueError("het_reference_distance must be > 0")

        self.reference_power = reference_power
        self.path_loss_exponent = path_loss_exponent
        self.noise_std = noise_std
        self.reference_distance = reference_distance
        self.het_factor = het_factor
        self.het_reference_distance = het_reference_distance
        self.rng = rng

        self.P0 = reference_power
        self.n = path_loss_exponent
        self.sigma = noise_std
        self.d0 = reference_distance

    def noise_std_at(self, distance):
        """Per-anchor noise std-dev at range `distance` (scalar or array).

        Single source of truth for the (possibly distance-dependent) sigma:
        both `rssi()` here and the CRLB must use the SAME sigma per anchor, so
        anything that needs the per-anchor sigma should call this rather than
        re-deriving the formula. With het_factor=0 this is just `noise_std`.
        """
        distance = np.asarray(distance, dtype=float)
        return self.noise_std * (1.0 + self.het_factor * distance / self.het_reference_distance)

    def _draw_noise(self, sigma):
        # Use the caller-supplied seeded stream if given (reproducible runs),
        # else fall back to the legacy global np.random state.
        if self.rng is not None:
            return self.rng.normal(0.0, sigma)
        return np.random.normal(0, sigma)

    def rssi(self, distance):
        distance = max(distance, np.finfo(float).eps)
        sigma = self.noise_std_at(distance)
        noise = self._draw_noise(sigma)
        path_loss = 10 * self.path_loss_exponent * np.log10(
            distance / self.reference_distance
        )
        return self.reference_power - path_loss + noise

    def rssi_matrix(self, anchors, targets):
        rssi_values = np.zeros((len(anchors), len(targets)))

        for anchor_index, anchor in enumerate(anchors):
            for target_index, target in enumerate(targets):
                distance = np.linalg.norm(anchor - target)
                rssi_values[anchor_index, target_index] = self.rssi(distance)

        return rssi_values
