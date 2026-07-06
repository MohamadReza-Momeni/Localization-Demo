import numpy as np
from .base_solver import BaseSolver


class ParticleFilterSolver(BaseSolver):
    def __init__(self, num_particles=2000, num_iterations=15, map_bounds=(0, 1000)):
        """
        :param num_particles: How many 'guesses' to scatter on the map.
        :param num_iterations: How many times to filter and resample them.
        :param map_bounds: The min and max coordinates of the area.
        """
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.bounds = map_bounds

    def solve(self, anchors, distances, x0=None):
        anchors = np.asarray(anchors)
        distances = np.asarray(distances)
        
        # 1. INITIALIZATION: Scatter particles
        if x0 is not None:
            # WARM START: If we have a rough guess, scatter particles in a wide 
            # 150-meter radius normal distribution around it to save time.
            particles = np.random.normal(
                loc=x0, scale=150.0, size=(self.num_particles, 2)
            )
        else:
            # COLD START: Scatter uniformly across the entire map
            particles = np.random.uniform(
                self.bounds[0], self.bounds[1], size=(self.num_particles, 2)
            )

        # Ensure particles don't spawn outside the map
        particles = np.clip(particles, self.bounds[0], self.bounds[1])

        # Core Particle Filter Loop
        for iteration in range(self.num_iterations):
            
            # A. PREDICT (Add random walk/jitter to explore the space)
            # Jitter shrinks as iterations go on to "zoom in" on the target
            jitter_scale = max(5.0, 50.0 / (iteration + 1))
            particles += np.random.normal(0, jitter_scale, size=particles.shape)
            particles = np.clip(particles, self.bounds[0], self.bounds[1])

            # B. UPDATE (Score the particles)
            # Calculate the distance from EVERY particle to EVERY anchor at once using vectorized numpy
            # diff shape: (num_particles, num_anchors, 2)
            diff = particles[:, np.newaxis, :] - anchors[np.newaxis, :, :]
            # est_distances shape: (num_particles, num_anchors)
            est_distances = np.linalg.norm(diff, axis=2)

            # Calculate the error between predicted distances and measured distances
            errors = np.abs(est_distances - distances)
            
            # Calculate weights based on how small the error is (using a Gaussian likelihood function)
            # We use a loose sigma (e.g., 20) because we know the measurements are highly noisy
            measurement_sigma = 20.0 
            weights = np.exp(-np.sum(errors**2, axis=1) / (2 * measurement_sigma**2))

            # Handle edge case: if all particles are so bad the weight is 0
            weight_sum = np.sum(weights)
            if weight_sum == 0 or np.isnan(weight_sum):
                weights = np.ones(self.num_particles) / self.num_particles
            else:
                weights /= weight_sum  # Normalize so all weights add up to 1.0

            # C. RESAMPLE (Survival of the fittest)
            # Pick particles based on their weight probabilities (good ones get duplicated, bad ones die)
            indices = np.random.choice(
                self.num_particles, size=self.num_particles, p=weights, replace=True
            )
            particles = particles[indices]

        # 3. ESTIMATE: Final position is the average of the surviving particles
        final_estimate = np.mean(particles, axis=0)

        return {
            "solution": final_estimate,
            "success": True, # Particle filters virtually never "fail" to run
        }