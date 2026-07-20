"""
src/scenario/clustered_anchors.py

E) Anchor-clustering scenario generator.

The professor's note ("3 scenarios, 20m-40m clustering") is a geometry
study, separate from the P0/beta/sigma signal-parameter sweep in
sweep_config.py/sweep_task.py: instead of asking "how does noise/PLE
affect error", it asks "how does anchor DENSITY/SPACING affect error,
holding the signal model fixed". This matters for localization because the
CRLB (crlb.py) already tells us the achievable bound depends on anchor
geometry (GDOP) as well as beta/sigma — tightly clustered anchors are
close to collinear-in-effect from the target's point of view, which
degrades the geometric dilution of precision even with perfect signal
conditions.

--- Model ---
`n_clusters` cluster centers are spread across the simulation area (on an
approximately even grid with jitter, so clusters don't overlap by
construction). `anchor_count` anchors are then divided as evenly as
possible across the clusters, and each anchor is placed at
`cluster_center + independent uniform offset in [-cluster_spacing/2, +cluster_spacing/2]`
on both axes — so `cluster_spacing` is (in expectation) the typical
distance between two anchors in the same cluster, i.e. the "how tight is
the cluster" knob the professor's note is asking to sweep (20m / 30m / 40m
are three natural scenario values, but any spacing works).

This is intentionally a NEW class (not a modification of the existing
AnchorGenerator, which this project doesn't have the source of on hand to
edit safely) with the same generate()-returns-ndarray interface used
everywhere else (AnchorGenerator(...).generate()), so it's a drop-in
alternative anywhere an anchor layout is needed:

    anchors = ClusteredAnchorGenerator(
        anchor_count=6, x_range=(0, 1000), y_range=(0, 1000),
        seed=None, n_clusters=2, cluster_spacing=30.0,
    ).generate()

--- Degenerate cases worth knowing about ---
- cluster_spacing=0 collapses every anchor in a cluster onto the same
  point -> that cluster contributes a rank-deficient block to the Fisher
  Information Matrix (crlb.py already detects and reports this via
  np.linalg.cond -> np.inf, it isn't a bug in either module).
- n_clusters > anchor_count is invalid (can't have more clusters than
  anchors) and raises ValueError.
- If a cluster's jittered spread would push anchors outside x_range/y_range,
  anchors are clipped back into bounds. Clipping a very tight cluster near
  a boundary corner can slightly reduce the EFFECTIVE spacing between
  anchors in that cluster relative to `cluster_spacing` — expected, not a bug.
"""
import numpy as np


class ClusteredAnchorGenerator:
    def __init__(self, anchor_count, x_range, y_range, seed=None,
                 n_clusters=3, cluster_spacing=30.0):
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if n_clusters > anchor_count:
            raise ValueError(
                f"n_clusters ({n_clusters}) cannot exceed anchor_count ({anchor_count})"
            )
        if cluster_spacing < 0:
            raise ValueError("cluster_spacing must be >= 0")

        self.anchor_count = anchor_count
        self.x_range = x_range
        self.y_range = y_range
        self.n_clusters = n_clusters
        self.cluster_spacing = cluster_spacing
        self.rng = np.random.default_rng(seed)

    def _cluster_centers(self):
        """Spread n_clusters centers roughly evenly across the area on a
        near-square grid, then jitter each within its grid cell so centers
        aren't perfectly regular (avoids accidentally-degenerate symmetric
        geometries) while staying well-separated from each other."""
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range

        n_cols = int(np.ceil(np.sqrt(self.n_clusters)))
        n_rows = int(np.ceil(self.n_clusters / n_cols))

        cell_w = (x_max - x_min) / n_cols
        cell_h = (y_max - y_min) / n_rows

        centers = []
        for i in range(self.n_clusters):
            row, col = divmod(i, n_cols)
            cell_x_min = x_min + col * cell_w
            cell_y_min = y_min + row * cell_h
            # Center of the cell, jittered by up to 20% of the cell size so
            # cluster centers aren't perfectly regular.
            cx = cell_x_min + cell_w / 2 + self.rng.uniform(-0.2, 0.2) * cell_w
            cy = cell_y_min + cell_h / 2 + self.rng.uniform(-0.2, 0.2) * cell_h
            centers.append([np.clip(cx, x_min, x_max), np.clip(cy, y_min, y_max)])

        return np.array(centers)

    def generate(self):
        centers = self._cluster_centers()

        # Divide anchors across clusters as evenly as possible (round-robin
        # for any remainder) rather than requiring anchor_count % n_clusters == 0.
        base, remainder = divmod(self.anchor_count, self.n_clusters)
        counts = [base + (1 if i < remainder else 0) for i in range(self.n_clusters)]

        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        half_spacing = self.cluster_spacing / 2.0

        anchors = []
        for center, count in zip(centers, counts):
            offsets = self.rng.uniform(-half_spacing, half_spacing, size=(count, 2))
            cluster_anchors = center + offsets
            cluster_anchors[:, 0] = np.clip(cluster_anchors[:, 0], x_min, x_max)
            cluster_anchors[:, 1] = np.clip(cluster_anchors[:, 1], y_min, y_max)
            anchors.append(cluster_anchors)

        return np.vstack(anchors)
