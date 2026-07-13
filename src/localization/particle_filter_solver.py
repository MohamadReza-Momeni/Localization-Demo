import numpy as np
from .base_solver import BaseSolver


class ParticleFilterSolver(BaseSolver):
    def __init__(self, num_particles=2000, num_iterations=15, x_bounds=(0, 1000), y_bounds=(0, 1000),
                 measurement_sigma=75.0):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.measurement_sigma = measurement_sigma

    def solve(self, anchors, distances, x0=None):
        anchors = np.asarray(anchors)
        distances = np.asarray(distances)

        if x0 is not None:
            particles = np.random.normal(loc=x0, scale=150.0, size=(self.num_particles, 2))
        else:
            # UPDATED: Scatter using dynamic bounds
            x_parts = np.random.uniform(self.x_bounds[0], self.x_bounds[1], self.num_particles)
            y_parts = np.random.uniform(self.y_bounds[0], self.y_bounds[1], self.num_particles)
            particles = np.column_stack((x_parts, y_parts))

        # UPDATED: Clip X and Y independently
        particles[:, 0] = np.clip(particles[:, 0], self.x_bounds[0], self.x_bounds[1])
        particles[:, 1] = np.clip(particles[:, 1], self.y_bounds[0], self.y_bounds[1])

        for iteration in range(self.num_iterations):
            jitter_scale = max(5.0, 50.0 / (iteration + 1))
            particles += np.random.normal(0, jitter_scale, size=particles.shape)

            # UPDATED: Clip X and Y independently during iteration
            particles[:, 0] = np.clip(particles[:, 0], self.x_bounds[0], self.x_bounds[1])
            particles[:, 1] = np.clip(particles[:, 1], self.y_bounds[0], self.y_bounds[1])

            diff = particles[:, np.newaxis, :] - anchors[np.newaxis, :, :]
            est_distances = np.linalg.norm(diff, axis=2)
            errors = np.abs(est_distances - distances)

            measurement_sigma = self.measurement_sigma
            weights = np.exp(-np.sum(errors ** 2, axis=1) / (2 * measurement_sigma ** 2))

            weight_sum = np.sum(weights)
            if weight_sum == 0 or np.isnan(weight_sum):
                weights = np.ones(self.num_particles) / self.num_particles
            else:
                weights /= weight_sum

            indices = np.random.choice(self.num_particles, size=self.num_particles, p=weights, replace=True)
            particles = particles[indices]

        final_estimate = np.mean(particles, axis=0)

        return {"solution": final_estimate, "success": True}