import concurrent.futures
import pandas as pd
from src.experiments.task import SimulationTask


class BatchExecutor:
    """SRP: Responsible ONLY for distributing simulation tasks across CPU cores."""

    def __init__(self, task: SimulationTask):
        self.task = task

    def run(self, run_count: int, max_workers: int = None, use_threads: bool = False) -> pd.DataFrame:
        """
        use_threads: set True when calling from an environment where re-importing the
        entry-point module is unsafe (e.g. Streamlit apps on Windows, where the
        top-level script runs Streamlit UI calls unconditionally on import and
        ProcessPoolExecutor's spawn-based workers would re-trigger them). Threads avoid
        that entirely since they share the same process/interpreter. numpy/scipy/cyipopt
        release the GIL during their heavy C-level computation, so real concurrency is
        still achieved — just not perfectly linear multi-core scaling like processes give.
        """
        all_rows = []

        executor_cls = (
            concurrent.futures.ThreadPoolExecutor
            if use_threads
            else concurrent.futures.ProcessPoolExecutor
        )

        with executor_cls(max_workers=max_workers) as executor:
            # Map the execute method of our task to the range of run IDs
            results_generator = executor.map(self.task.execute, range(run_count))

            # Aggregate the results as they finish
            for run_rows in results_generator:
                all_rows.extend(run_rows)

        return pd.DataFrame(all_rows)