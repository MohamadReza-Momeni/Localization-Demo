import numpy as np

def _scenario_seed(anchors, x_range, y_range) -> int:
    """Stable 32-bit seed derived from the scenario geometry."""
    key = np.concatenate([
        np.asarray(anchors, dtype=float).ravel(),
        np.asarray([x_range[0], x_range[1], y_range[0], y_range[1]], dtype=float),
    ])
    return int(abs(hash(key.tobytes()))) & 0xFFFFFFFF

def generate_starting_points(anchors, x0, x_range, y_range,
                             labels=("warm_start", "anchor_centroid", "random_point"),
                             seed=None):
    anchors = np.asarray(anchors, dtype=float)
    points = {}

    if "warm_start" in labels:
        if x0 is not None:
            points["warm_start"] = np.asarray(x0, dtype=float)
        else:
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