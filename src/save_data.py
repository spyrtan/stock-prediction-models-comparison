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
    Save raw time series data (from Yahoo Finance) to a CSV file.
    """
    if isinstance(data, pd.Series):
        df = data.to_frame(name="Close")
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Unsupported data format for saving raw data.")

    file_name = f"{ticker}_raw.csv"
    file_path = os.path.join(RAW_DIR, file_name)
    df.to_csv(file_path)

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
