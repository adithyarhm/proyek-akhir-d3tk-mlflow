import pandas as pd
from src.config import DATA_PATH

def load_data():
    """
    Ini untuk load data
    """
    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df