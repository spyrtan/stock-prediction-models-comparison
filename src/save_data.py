import os
import pandas as pd
import numpy as np

# Define the root directory of the project dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Ensure required directories exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def save_raw_data(data, ticker):
    """
    Save cleaned raw time series data (from Yahoo Finance) to a CSV file.
    Only 'Date' and 'Close' columns are retained. Strips MultiIndex and non-numeric rows.
    """
    if isinstance(data, pd.Series):
        df = data.to_frame(name="Close")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Unsupported data format for saving raw data.")

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)

    # Reset index if necessary
    if df.index.name == "Date" or df.index.name is None:
        df = df.reset_index()

    # Keep only 'Date' and 'Close', force numeric type, drop NaNs
    if "Date" in df.columns and "Close" in df.columns:
        df = df[["Date", "Close"]]
        df = df[pd.to_numeric(df["Close"], errors="coerce").notnull()]
        df["Close"] = df["Close"].astype(float)
    else:
        raise ValueError("Expected columns 'Date' and 'Close' not found in data.")

    df = df.dropna()

    file_name = f"{ticker}_raw.csv"
    file_path = os.path.join(RAW_DIR, file_name)
    df.to_csv(file_path, index=False)

    print("✅ Raw data saved!")
    print(f"📄 File name: {file_name}")
    print(f"📂 Full path: {file_path}")

def save_processed_data(X_train, y_train, X_test, y_test, ticker):
    """
    Save processed training and testing datasets as .npy files.
    """
    ticker_dir = os.path.join(PROCESSED_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)

    np.save(os.path.join(ticker_dir, "X_train.npy"), X_train)
    np.save(os.path.join(ticker_dir, "y_train.npy"), y_train)
    np.save(os.path.join(ticker_dir, "X_test.npy"), X_test)
    np.save(os.path.join(ticker_dir, "y_test.npy"), y_test)

    print(f"✅ Processed data saved to {ticker_dir}")
