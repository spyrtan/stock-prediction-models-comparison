import pandas as pd
import os
from pathlib import Path

def load_data(ticker, directory="data"):
    """
    Load historical price data for the specified ticker from a local CSV file.

    Args:
        ticker (str): Stock ticker symbol (e.g. 'AAPL')
        directory (str): Directory where the CSV file is located

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame
    """
    base_dir = Path(__file__).resolve().parent.parent  # points to project root
    filepath = base_dir / directory / f"{ticker}_prices.csv"

    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} not found.")

    df = pd.read_csv(filepath)
    print(f"📁 Data loaded from {filepath}")
    return df
