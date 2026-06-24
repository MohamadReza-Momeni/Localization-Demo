import numpy as np


def rssi_to_distance(rssi, P0, n, d0=1.0):
    return d0 * 10 ** ((P0 - rssi) / (10 * n))