from pathlib import Path
import pandas as pd

def load_data(clean: bool = True):
    """
    Loads the YouTube Trending dataset from the data/raw folder.
    
    Args:
        clean (bool): if True, attempts to read the cleaned Parquet file first.
                      if False or missing, reads the original CSV.

    Returns:
        pd.DataFrame: the loaded dataset
    """
    base = Path(__file__).resolve().parent.parent  # go up from src/
    raw_path = base / "data" / "raw"

    parquet_path = raw_path / "USvideos_clean.parquet"
    csv_path = raw_path / "USvideos.csv"

    if clean and parquet_path.exists():
        print("Loading cleaned parquet file...")
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        print("Loading raw CSV file...")
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError("Dataset not found in data/raw/.")
