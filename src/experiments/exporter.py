import pandas as pd

class ResultExporter:
    """SRP: Responsible ONLY for data persistence (saving results)."""
    
    @staticmethod
    def save_csv(df: pd.DataFrame, filename: str = "results.csv"):
        df.to_csv(filename, index=False)