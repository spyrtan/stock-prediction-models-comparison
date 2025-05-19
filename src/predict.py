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
        print("⚠️ Failed to fetch data — using local raw backup.")
        raw_path = os.path.join("data", "raw", f"{ticker}_raw.csv")
        df = pd.read_csv(raw_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df = df[["Date", "Close"]]
        df = df[pd.to_numeric(df["Close"], errors="coerce").notnull()]
        df["Close"] = df["Close"].astype(float)

    save_data.save_raw_data(df, ticker)

    close_prices = df["Close"].values.reshape(-1, 1)

    # Prepare latest input for prediction
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
