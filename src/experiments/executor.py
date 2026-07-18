import concurrent.futures
import itertools
import pandas as pd
from src.experiments.task import SimulationTask

class BatchExecutor:
    """SRP: Responsible ONLY for distributing simulation tasks across CPU cores."""
    
    def __init__(self, task: SimulationTask):
        self.task = task

    def run(self, run_count: int, max_workers: int = None) -> pd.DataFrame:
        all_rows = []
        
        # Generate the Cartesian product of all parameters (The Grid Sweep)
        # This creates a flat list of tuples: (run_id, p0, ple, sigma)
        # e.g., [(0, -40, 2.2, 1.0), (0, -40, 2.2, 2.0), ..., (99, -30, 4.0, 5.0)]
        job_parameters = list(itertools.product(
            range(run_count),
            self.task.config.p0_values,
            self.task.config.ple_values,
            self.task.config.noise_values
        ))
        
        # Distribute the workload across multiple processes
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # We map our newly generated flat list of combinations directly into the task
            results_generator = executor.map(self.task.execute, job_parameters)
            
            # Aggregate the results as they finish
            for run_rows in results_generator:
                all_rows.extend(run_rows)
                
        return pd.DataFrame(all_rows)