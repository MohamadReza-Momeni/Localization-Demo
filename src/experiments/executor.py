import concurrent.futures
import itertools
import pandas as pd
from src.experiments.task import SimulationTask

class BatchExecutor:
    """SRP: Responsible ONLY for distributing simulation tasks across CPU cores."""
    
    def __init__(self, task: SimulationTask):
        self.task = task

    def run(self, run_count: int, max_workers: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        all_solver_rows = []
        all_measurement_rows = []
        
        job_parameters = list(itertools.product(
            range(run_count),
            self.task.config.p0_values,
            self.task.config.ple_values,
            self.task.config.noise_values
        ))
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results_generator = executor.map(self.task.execute, job_parameters)
            
            # Unpacks the two lists coming from the task
            for solver_rows, measurement_rows in results_generator:
                all_solver_rows.extend(solver_rows)
                all_measurement_rows.extend(measurement_rows)
                
        # for params in job_parameters:
        #     solver_rows, measurement_rows = self.task.execute(params)
        #     all_solver_rows.extend(solver_rows)
        #     all_measurement_rows.extend(measurement_rows)
        # Returns TWO DataFrames back to app.py
        return pd.DataFrame(all_solver_rows), pd.DataFrame(all_measurement_rows)