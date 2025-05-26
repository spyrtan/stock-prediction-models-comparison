import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from src import preprocess, save_data
from tensorflow.keras.models import load_model
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

def predict_next(model_name: str, ticker: str) -> float:
    """
    Fetches the latest stock data, prepares input, loads the specified model, and returns the prediction.
    """
    try:
        print(f"📡 Downloading last 5 years of data for {ticker}...")
        end = datetime.today()
        start = end - timedelta(days=5 * 365)
        df = pd.read_csv(
            f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={int(start.timestamp())}&period2={int(end.timestamp())}&interval=1d&events=history&includeAdjustedClose=true"
        )
        df = df[["Date", "Close"]]
        df = df[pd.to_numeric(df["Close"], errors="coerce").notnull()]
        df["Close"] = df["Close"].astype(float)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        print("✅ Data successfully fetched from the internet.")
    except:
        print("⚠️ Failed to fetch data — using local backup.")
        raw_path = os.path.join("data", "raw", f"{ticker}_raw.csv")
        df = pd.read_csv(raw_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df = df[["Date", "Close"]]
        df = df[pd.to_numeric(df["Close"], errors="coerce").notnull()]
        df["Close"] = df["Close"].astype(float)

    save_data.save_raw_data(df, ticker)

    close_prices = df["Close"].values.reshape(-1, 1)

    # Prepare the latest input for prediction
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    window_size = 30
    if len(scaled_data) < window_size:
        raise ValueError("Not enough data for prediction.")

    X_latest = np.array([scaled_data[-window_size:]])

    model_dir = "models"
    pred_scaled = None

    if model_name == "LSTM":
        model = load_model(os.path.join(model_dir, f"{ticker}_lstm_model.h5"))
        pred_scaled = model.predict(X_latest)

    elif model_name == "CNN":
        model = load_model(os.path.join(model_dir, f"{ticker}_cnn_model.h5"))
        pred_scaled = model.predict(X_latest)

    elif model_name == "Transformer":
        model = load_model(os.path.join(model_dir, f"{ticker}_transformer_model.keras"))
        pred_scaled = model.predict(X_latest)

    elif model_name == "XGBoost":
        model = xgb.XGBRegressor()
        model.load_model(os.path.join(model_dir, f"{ticker}_xgboost_model.json"))
        X_latest_flat = X_latest.reshape(X_latest.shape[0], -1)
        pred_scaled = model.predict(X_latest_flat).reshape(-1, 1)

    elif model_name == "ARIMA":
        series = df["Close"].dropna()
        model = ARIMA(series, order=(5, 1, 0))
        model_fit = model.fit()
        pred_value = model_fit.forecast(steps=1)[0]
        print(f"📈 Forecast ({model_name}): {pred_value:.2f}")
        return float(pred_value)

    else:
        raise ValueError("Unknown model name.")

    pred_value = scaler.inverse_transform(pred_scaled)[0][0]
    print(f"📈 Forecast ({model_name}): {pred_value:.2f}")
    return float(pred_value)

def save_prediction(ticker: str, model_name: str, predicted_value: float):
    """
    Saves the prediction to a CSV file named {model}_predictions.csv inside results/{ticker}/.
    Each prediction is saved for the next available trading day after the last known date.
    """
    result_dir = os.path.join("results", ticker)
    os.makedirs(result_dir, exist_ok=True)

    # Load the last known date from local raw data
    raw_path = os.path.join("data", "raw", f"{ticker}_raw.csv")
    df = pd.read_csv(raw_path)
    df["Date"] = pd.to_datetime(df["Date"])
    last_known_date = df["Date"].max()

    # Compute the next trading day (skip weekends)
    next_day = last_known_date + timedelta(days=1)
    while next_day.weekday() > 4:  # 5=Saturday, 6=Sunday
        next_day += timedelta(days=1)
    prediction_date = next_day.strftime("%Y-%m-%d")

    result_file = os.path.join(result_dir, f"{model_name.lower()}_predictions.csv")

    if os.path.exists(result_file):
        df_pred = pd.read_csv(result_file)
    else:
        df_pred = pd.DataFrame(columns=["Date", "Prediction"])

    # Overwrite any existing entry for that date
    df_pred = df_pred[df_pred["Date"] != prediction_date]
    df_pred = pd.concat([df_pred, pd.DataFrame([{"Date": prediction_date, "Prediction": predicted_value}])], ignore_index=True)

    df_pred.sort_values("Date", inplace=True)
    df_pred.to_csv(result_file, index=False)

    return result_file

def update_actuals(ticker: str, model_name: str):
    """
    Updates the actual closing prices for previously predicted dates and calculates prediction error.
    """
    import yfinance as yf

    result_file = os.path.join("results", ticker, f"{model_name.lower()}_predictions.csv")
    if not os.path.exists(result_file):
        print("❌ Prediction file not found.")
        return

    df = pd.read_csv(result_file)
    if "Actual" not in df.columns:
        df["Actual"] = np.nan

    print(f"📥 Updating actual closing prices for {ticker}...")

    updated_count = 0

    for i, row in df.iterrows():
        date = row["Date"]
        if not pd.isna(row["Actual"]):
            continue

        try:
            hist = yf.download(ticker, start=date, end=date, interval="1d", progress=False)
            if not hist.empty:
                actual_close = hist["Close"].iloc[0]
                df.at[i, "Actual"] = actual_close
                updated_count += 1
                print(f"✅ {date}: {actual_close}")
            else:
                print(f"⚠️ No data for {date}")
        except Exception as e:
            print(f"❌ Error fetching data for {date}: {e}")

    # Compute prediction error (Prediction - Actual)
    if "Error" not in df.columns:
        df["Error"] = np.nan

    df["Error"] = df.apply(
        lambda row: row["Prediction"] - row["Actual"]
        if not pd.isna(row["Prediction"]) and not pd.isna(row["Actual"]) else np.nan,
        axis=1
    )

    df.to_csv(result_file, index=False)
    print(f"\n💾 File updated: {result_file}")
    if updated_count == 0:
        print("ℹ️ No actual values were updated. Try again later.")
