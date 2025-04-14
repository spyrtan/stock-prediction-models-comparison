import os
from pathlib import Path
from src import preprocess
from src import save_data
from src.load_data import load_data
import pandas as pd
import numpy as np

def main():
    print("=== Stock Data Loader ===")
    print("[1] Load data from raw CSV")
    print("[2] Download data from Yahoo Finance")
    choice = input("🔢 Your choice: ")

    BASE_DIR = Path(__file__).resolve().parent
    RAW_DATA_DIR = BASE_DIR / "data" / "raw"

    if choice == "1":
        ticker = input("📈 Enter stock ticker (e.g. AAPL): ").upper()
        try:
            df = load_data(ticker)
            print(f"\n✅ Successfully loaded raw data for {ticker}")
            print(df.head())
        except Exception as e:
            print(f"\n❌ Error occurred while loading CSV: {e}")

    elif choice == "2":
        print("\n=== Stock Data Download & Preparation ===\n")
        ticker = input("📈 Enter stock ticker (e.g. AAPL): ").upper()
        start = input("📅 Start date (YYYY-MM-DD): ")
        end = input("📅 End date (YYYY-MM-DD): ")
        interval = input("⏱️ Interval (e.g. 1d, 1wk, 1mo): ")
        window_size = int(input("🔁 Window size (e.g. 30): "))

        try:
            X_train, y_train, X_test, y_test, scaler, series = preprocess.prepare_data(
                ticker=ticker,
                start=start,
                end=end,
                interval=interval,
                window_size=window_size
            )

            print("\n✅ Data successfully downloaded and prepared.")
            print(f"🔹 X_train shape: {X_train.shape}")
            print(f"🔹 y_train shape: {y_train.shape}")
            print(f"🔹 X_test shape: {X_test.shape}")
            print(f"🔹 y_test shape: {y_test.shape}")

            # Zapisz dane raw
            if isinstance(series, pd.Series):
                df = series.to_frame(name="Close")
            elif isinstance(series, pd.DataFrame):
                df = series.copy()
            elif isinstance(series, np.ndarray):
                df = pd.DataFrame(series, columns=["Close"])
            else:
                raise ValueError(f"Unsupported data format for saving raw data. Type: {type(series)}")

            if df.empty:
                print("❌ No data to save. Raw data is empty.")
            else:
                RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
                file_name = f"{ticker}_raw.csv"
                full_path = RAW_DATA_DIR / file_name
                df.to_csv(full_path, index=False)
                print("✅ Raw data saved!")
                print(f"📄 File name: {file_name}")
                print(f"📂 Full path: {full_path}")

            # Zapisz dane przetworzone
            save_data.save_processed_data(X_train, y_train, X_test, y_test, ticker)

        except Exception as e:
            print(f"\n❌ Error occurred: {e}")

    else:
        print("⚠️ Invalid choice. Please select 1 or 2.")

if __name__ == "__main__":
    main()
