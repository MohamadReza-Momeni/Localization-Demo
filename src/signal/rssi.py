import numpy as np


class RSSIModel:
    def __init__(
        self,
        reference_power=-40,
        path_loss_exponent=2.2,
        noise_std=2.0,
        reference_distance=1.0,
    ):
        self.reference_power = reference_power
        self.path_loss_exponent = path_loss_exponent
        self.noise_std = noise_std
        self.reference_distance = reference_distance

        self.P0 = reference_power
        self.n = path_loss_exponent
        self.sigma = noise_std
        self.d0 = reference_distance

    def rssi(self, distance):
        distance = max(distance, np.finfo(float).eps)
        noise = np.random.normal(0, self.noise_std)
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