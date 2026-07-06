import concurrent.futures
import pandas as pd
from src.experiments.task import SimulationTask

class BatchExecutor:
    """SRP: Responsible ONLY for distributing simulation tasks across CPU cores."""
    
    def __init__(self, task: SimulationTask):
        self.task = task

    def run(self, run_count: int, max_workers: int = None) -> pd.DataFrame:
        all_rows = []
        
        # Distribute the workload across multiple processes
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map the execute method of our task to the range of run IDs
            results_generator = executor.map(self.task.execute, range(run_count))
            
            # Aggregate the results as they finish
            for run_rows in results_generator:
                all_rows.extend(run_rows)
                
        return pd.DataFrame(all_rows)