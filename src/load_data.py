import pandas as pd
import os
from pathlib import Path

def load_data(ticker, directory="data"):
    base_dir = Path(__file__).resolve().parent.parent  # <- src/
    filepath = base_dir / directory / f"{ticker}_prices.csv"

    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} not found.")

    df = pd.read_csv(filepath)
    print(f"📁 Data loaded from {filepath}")
    return df
