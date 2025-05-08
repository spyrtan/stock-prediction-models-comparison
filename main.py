import os
from pathlib import Path
from src import preprocess
from src import save_data
from src.load_data import load_data
import pandas as pd
import numpy as np
import subprocess
import sys

def ensure_data_exists(ticker):
    processed_dir = Path("data") / "processed" / ticker
    required_files = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]
    if not all((processed_dir / f).exists() for f in required_files):
        print(f"\n⚠️ Dane dla {ticker} nie są gotowe.")
        method = input("📥 Wczytać dane z pliku [C] czy pobrać z internetu [Y]? ").strip().upper()
        if method == "C":
            try:
                df = load_data(ticker)
                print(f"✅ Dane z pliku CSV dla {ticker} załadowane.")
                print(df.head())
            except Exception as e:
                print(f"❌ Błąd: {e}")
                sys.exit(1)
        elif method == "Y":
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
                save_data.save_raw_data(series, ticker)
                save_data.save_processed_data(X_train, y_train, X_test, y_test, ticker)
            except Exception as e:
                print(f"❌ Błąd: {e}")
                sys.exit(1)
        else:
            print("❌ Niepoprawny wybór. Kończę.")
            sys.exit(1)


def main():
    print("=== Stock Project CLI ===")
    print("[1] Train all models")
    print("[2] Evaluate all models")
    print("[3] Predict using selected model")
    choice = input("🔢 Your choice: ")

    BASE_DIR = Path(__file__).resolve().parent
    ticker = input("📈 Enter stock ticker (e.g. AAPL): ").upper()
    os.environ["TICKER"] = ticker

    if choice == "1":
        ensure_data_exists(ticker)
        print("\n🚀 Training all models...\n")
        subprocess.run([sys.executable, str(BASE_DIR / "src" / "train.py")])

    elif choice == "2":
        ensure_data_exists(ticker)
        print("\n📊 Evaluating all models...\n")
        subprocess.run([sys.executable, str(BASE_DIR / "src" / "evaluate.py")])

    elif choice == "3":
        ensure_data_exists(ticker)
        print("\n🔮 Predykcja (do zrobienia)...")
        # tutaj później dodamy kod do predykcji

    else:
        print("⚠️ Invalid choice. Please select 1, 2 or 3.")

if __name__ == "__main__":
    main()
