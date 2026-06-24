import numpy as np

def euclidean_error(true, est):
    return np.linalg.norm(true - est)