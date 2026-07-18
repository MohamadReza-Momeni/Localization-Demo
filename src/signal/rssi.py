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

    def rssi(self, distance, samples=1):
        distance = max(distance, np.finfo(float).eps)
        
        # Draw multiple noise measurements
        noises = np.random.normal(0, self.noise_std, samples)
        
        path_loss = 10 * self.path_loss_exponent * np.log10(
            distance / self.reference_distance
        )
        
        base_rssi = self.reference_power - path_loss
        raw_rssis = base_rssi + noises
        
        # Return both the averaged value (for the solvers) and the raw array
        return np.mean(raw_rssis), raw_rssis

    def rssi_matrix(self, anchors, targets, samples=1):
        rssi_values = np.zeros((len(anchors), len(targets)))
        # NEW: A 3D matrix to hold the raw samples for the new table
        raw_samples = np.zeros((len(anchors), len(targets), samples))

        for anchor_index, anchor in enumerate(anchors):
            for target_index, target in enumerate(targets):
                distance = np.linalg.norm(anchor - target)
                mean_rssi, raw_rssi_array = self.rssi(distance, samples)
                
                rssi_values[anchor_index, target_index] = mean_rssi
                raw_samples[anchor_index, target_index] = raw_rssi_array

        return rssi_values, raw_samples