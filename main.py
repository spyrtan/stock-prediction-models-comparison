import os
from pathlib import Path
from datetime import datetime
from src import preprocess
from src import save_data
from src.load_data import load_data
from src.predict import predict_next, save_prediction, update_actuals
import pandas as pd
import numpy as np
import subprocess
import sys

def ensure_data_exists(ticker):
    processed_dir = Path("data") / "processed" / ticker
    required_files = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]
    if not all((processed_dir / f).exists() for f in required_files):
        print(f"\n⚠️ Processed data for {ticker} not found.")
        method = input("📥 Load from CSV file [C] or fetch from the internet [Y]? ").strip().upper()
        if method == "C":
            try:
                df = load_data(ticker)
                print(f"✅ Data loaded from local CSV for {ticker}.")
                print(df.head())
            except Exception as e:
                print(f"❌ Error loading file: {e}")
                return False
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
                save_data.save_processed_data(X_train, y_train, X_test, y_test, scaler, ticker)
            except Exception as e:
                print(f"❌ Error while preparing data: {e}")
                return False
        else:
            print("❌ Invalid option. Returning to menu.")
            return False
    return True

def main():
    BASE_DIR = Path(__file__).resolve().parent

    while True:
        print("\n🧑‍🧠 === STOCK PREDICTION CLI ===")
        print("📅 Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        print("\n📁 Data management:")
        print(" [1] Download data        → Fetch last 5 years of data from Yahoo Finance")
        print(" [2] Train models         → Train all or selected models on processed data")
        print(" [3] Evaluate models      → Calculate MSE for each model")

        print("\n🔮 Prediction & analysis:")
        print(" [4] Make prediction      → Predict next day’s closing price")
        print(" [5] Update actual prices → Fill in real prices for saved predictions")

        print("\n❌ Exit:")
        print(" [Q] Quit program")

        choice = input("\n🔹 Your choice: ").strip().upper()

        if choice == "Q":
            print("👋 Exiting the program. Goodbye!")
            break

        if choice not in ["1", "2", "3", "4", "5"]:
            print("⚠️ Invalid option. Please choose between 1–5 or Q.")
            continue

        ticker = input("📈 Enter stock ticker (e.g. AAPL): ").upper()
        os.environ["TICKER"] = ticker

        if choice == "1":
            print(f"\n📦 Downloading {ticker} data for the last 5 years...")
            from datetime import timedelta
            import yfinance as yf

            end = datetime.today()
            start = end - timedelta(days=5 * 365)

            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d")

            if df.empty:
                print("❌ Failed to fetch data from Yahoo Finance.")
                continue

            save_data.save_raw_data(df, ticker)
            print("✅ Raw data saved to data/raw/")

            from src.preprocess import prepare_from_series
            close_prices = df["Close"].values.reshape(-1, 1)
            X_train, y_train, X_test, y_test, scaler = prepare_from_series(close_prices, window_size=90)
            save_data.save_processed_data(X_train, y_train, X_test, y_test, scaler, ticker)
            print("✅ Processed data saved to data/processed/")

        elif choice == "2":
            if ensure_data_exists(ticker):
                print("\n🎯 Select training mode:")
                print(" [1] Train all models")
                print(" [2] Train selected models only")

                sub_choice = input("🔢 Your choice: ").strip()
                if sub_choice not in ["1", "2"]:
                    print("⚠️ Invalid option.")
                    return

                try:
                    repeat = int(input("🔁 How many times should each model be trained? (default: 1): ") or "1")
                    if repeat < 1:
                        raise ValueError
                except ValueError:
                    print("⚠️ Invalid number. Using default value: 1")
                    repeat = 1

                os.environ["TRAIN_REPEAT"] = str(repeat)

                if sub_choice == "2":
                    print("\n🧑‍🧠 Select models to train (comma-separated):")
                    print(" [1] LSTM")
                    print(" [2] CNN")
                    print(" [3] Transformer")
                    print(" [4] XGBoost")
                    print(" [5] ARIMA")

                    selected = input("🔢 Your choices (e.g., 1,3,5): ").strip()

                    model_map = {
                        "1": "LSTM",
                        "2": "CNN",
                        "3": "Transformer",
                        "4": "XGBoost",
                        "5": "ARIMA"
                    }

                    selected_models = [model_map.get(c.strip()) for c in selected.split(",") if c.strip() in model_map]
                    if not selected_models:
                        print("⚠️ No valid models selected.")
                        return

                    os.environ["SELECTED_MODELS"] = ",".join(selected_models)
                else:
                    os.environ.pop("SELECTED_MODELS", None)

                subprocess.run([sys.executable, str(BASE_DIR / "src" / "train.py")], env=os.environ.copy())

        elif choice == "3":
            if ensure_data_exists(ticker):
                print("\n📊 Evaluating all models...\n")
                subprocess.run([sys.executable, str(BASE_DIR / "src" / "evaluate.py")])

        elif choice == "4":
            print("\n🤖 Choose prediction model:")
            print(" [1] LSTM         → Long Short-Term Memory")
            print(" [2] CNN          → Convolutional Neural Network")
            print(" [3] Transformer  → Attention-based model")
            print(" [4] XGBoost      → Gradient boosting")
            print(" [5] ARIMA        → Autoregressive model")
            print(" [A] All models")

            model_choice = input("🔢 Your choice: ").strip().upper()

            model_map = {
                "1": "LSTM",
                "2": "CNN",
                "3": "Transformer",
                "4": "XGBoost",
                "5": "ARIMA"
            }

            if model_choice == "A":
                predictions = {}
                for model_name in model_map.values():
                    try:
                        predicted_value = predict_next(model_name, ticker)
                        predictions[model_name] = predicted_value
                        print(f"\n🎯 Predicted closing price using {model_name}: {predicted_value:.2f}")
                    except Exception as e:
                        print(f"❌ {model_name} prediction error: {e}")

                save_all = input("\n💾 Save all predictions to files? (Y/N): ").strip().upper()
                if save_all == "Y":
                    for model_name, predicted_value in predictions.items():
                        try:
                            path = save_prediction(ticker, model_name, predicted_value)
                            print(f"✅ Prediction saved to: {path}")
                        except Exception as e:
                            print(f"❌ Error saving {model_name}: {e}")

            elif model_choice in model_map:
                model_name = model_map[model_choice]
                try:
                    predicted_value = predict_next(model_name, ticker)
                    print(f"\n🎯 Predicted closing price using {model_name}: {predicted_value:.2f}")
                    save = input("💾 Save prediction to file? (Y/N): ").strip().upper()
                    if save == "Y":
                        path = save_prediction(ticker, model_name, predicted_value)
                        print(f"✅ Prediction saved to: {path}")
                except Exception as e:
                    print(f"❌ Prediction error: {e}")
            else:
                print("❌ Invalid model selection.")

        elif choice == "5":
            print("\n📊 Select model to update actual prices:")
            print(" [1] LSTM")
            print(" [2] CNN")
            print(" [3] Transformer")
            print(" [4] XGBoost")
            print(" [5] ARIMA")
            print(" [A] All models")
            model_choice = input("🔢 Your choice: ").strip().upper()

            model_map = {
                "1": "LSTM",
                "2": "CNN",
                "3": "Transformer",
                "4": "XGBoost",
                "5": "ARIMA"
            }

            if model_choice == "A":
                for model_name in model_map.values():
                    print(f"\n🔁 Updating {model_name}...")
                    update_actuals(ticker, model_name)
            elif model_choice in model_map:
                model_name = model_map[model_choice]
                update_actuals(ticker, model_name)
            else:
                print("❌ Invalid model selection.")

if __name__ == "__main__":
    main()
