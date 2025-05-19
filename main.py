import os
from pathlib import Path
from src import preprocess
from src import save_data
from src.load_data import load_data
from src.predict import predict_next
import pandas as pd
import numpy as np
import subprocess
import sys

def ensure_data_exists(ticker):
    processed_dir = Path("data") / "processed" / ticker
    required_files = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]
    if not all((processed_dir / f).exists() for f in required_files):
        print(f"\n⚠️ Data for {ticker} is not ready.")
        method = input("📥 Load data from file [C] or fetch from the internet [Y]? ").strip().upper()
        if method == "C":
            try:
                df = load_data(ticker)
                print(f"✅ Data loaded from CSV file for {ticker}.")
                print(df.head())
            except Exception as e:
                print(f"❌ Error: {e}")
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
                print(f"❌ Error: {e}")
                sys.exit(1)
        else:
            print("❌ Invalid option. Exiting.")
            sys.exit(1)

def main():
    print("=== Stock Project CLI ===")
    print("[1] Download last 5 years of data and save as raw")
    print("[2] Train all models")
    print("[3] Evaluate all models")
    print("[4] Predict using selected model")
    choice = input("🔢 Your choice: ")

    BASE_DIR = Path(__file__).resolve().parent
    ticker = input("📈 Enter stock ticker (e.g. AAPL): ").upper()
    os.environ["TICKER"] = ticker

    if choice == "1":
        print(f"\n📦 Downloading {ticker} data for the last 5 years...")
        from datetime import datetime, timedelta
        import yfinance as yf

        end = datetime.today()
        start = end - timedelta(days=5*365)

        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d")

        if df.empty:
            print("❌ Failed to fetch data.")
            sys.exit(1)

        save_data.save_raw_data(df, ticker)
        print("✅ Data saved to data/raw/")

    elif choice == "2":
        ensure_data_exists(ticker)
        print("\n🚀 Training all models...\n")
        subprocess.run([sys.executable, str(BASE_DIR / "src" / "train.py")])

    elif choice == "3":
        ensure_data_exists(ticker)
        print("\n📊 Evaluating all models...\n")
        subprocess.run([sys.executable, str(BASE_DIR / "src" / "evaluate.py")])

    elif choice == "4":
        print("\n📊 Select model for prediction:")
        print("[1] LSTM")
        print("[2] CNN")
        print("[3] Transformer")
        print("[4] XGBoost")
        print("[5] ARIMA")
        model_choice = input("🔢 Your choice: ")

        model_map = {
            "1": "LSTM",
            "2": "CNN",
            "3": "Transformer",
            "4": "XGBoost",
            "5": "ARIMA"
        }

        if model_choice not in model_map:
            print("❌ Invalid model selection.")
            sys.exit(1)

        model_name = model_map[model_choice]

        try:
            predicted_value = predict_next(model_name, ticker)
            print(f"\n🎯 Predicted closing price for next day using {model_name}: {predicted_value:.2f}")
        except Exception as e:
            print(f"❌ Prediction error: {e}")

    else:
        print("⚠️ Invalid choice. Please select 1, 2, 3 or 4.")

if __name__ == "__main__":
    main()
