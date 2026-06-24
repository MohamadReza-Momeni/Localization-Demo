import numpy as np


def euclidean_error(true, est):
    return np.linalg.norm(np.array(true) - np.array(est))