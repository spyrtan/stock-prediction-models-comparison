import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from src import save_data
from tensorflow.keras.models import load_model
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

def get_next_trading_day(start_date):
    holidays_2025 = [
        "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
        "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25"
    ]
    holidays = pd.to_datetime(holidays_2025)
    date = start_date + timedelta(days=1)
    while date.weekday() >= 5 or date in holidays:
        date += timedelta(days=1)
    return date

def predict_next(model_name: str, ticker: str) -> float:
    print(f"📡 Downloading last 5 years of data for {ticker}...")
    try:
        end = datetime.today()
        start = end - timedelta(days=5 * 365)
        df = pd.read_csv(
            f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={int(start.timestamp())}&period2={int(end.timestamp())}&interval=1d&events=history&includeAdjustedClose=true"
        )
        df = df[["Date", "Close"]].dropna()
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.dropna().sort_values("Date")
        print("✅ Data successfully fetched from the internet.")
    except Exception:
        print("⚠️ Failed to fetch data — using local backup.")
        raw_path = os.path.join("data", "raw", f"{ticker}_raw.csv")
        df = pd.read_csv(raw_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df = df[["Date", "Close"]].dropna()
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    save_data.save_raw_data(df, ticker)

    close_prices = df["Close"].values.reshape(-1, 1)
    window_size = 30

    if len(close_prices) < window_size:
        raise ValueError("Not enough data for prediction.")

    model_dir = "models"

    if model_name == "ARIMA":
        series = df["Close"].dropna()
        model = ARIMA(series, order=(5, 1, 0))
        model_fit = model.fit()
        pred_value = model_fit.forecast(steps=1)[0]
        print(f"📈 Forecast ({model_name}): {pred_value:.2f}")
        return float(pred_value)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)
    X_latest = np.array([scaled_data[-window_size:]])

    if model_name == "LSTM":
        model = load_model(os.path.join(model_dir, f"{ticker}_lstm_model.keras"))
        pred_scaled = model.predict(X_latest)

    elif model_name == "CNN":
        model = load_model(os.path.join(model_dir, f"{ticker}_cnn_model.keras"))
        pred_scaled = model.predict(X_latest)

    elif model_name == "Transformer":
        model = load_model(os.path.join(model_dir, f"{ticker}_transformer_model.keras"))
        pred_scaled = model.predict(X_latest)

    elif model_name == "XGBoost":
        model = xgb.XGBRegressor()
        model.load_model(os.path.join(model_dir, f"{ticker}_xgboost_model.json"))
        X_flat = X_latest.reshape(X_latest.shape[0], -1)
        pred_scaled = model.predict(X_flat).reshape(-1, 1)

    else:
        raise ValueError("Unknown model name.")

    pred_value = scaler.inverse_transform(pred_scaled)[0][0]
    print(f"📈 Forecast ({model_name}): {pred_value:.2f}")
    return float(pred_value)

def save_prediction(ticker: str, model_name: str, predicted_value: float):
    result_dir = os.path.join("results", ticker)
    os.makedirs(result_dir, exist_ok=True)

    raw_path = os.path.join("data", "raw", f"{ticker}_raw.csv")
    df = pd.read_csv(raw_path)
    df["Date"] = pd.to_datetime(df["Date"])
    last_known_date = df["Date"].max()

    next_day = get_next_trading_day(last_known_date)
    prediction_date = next_day.strftime("%Y-%m-%d")

    result_file = os.path.join(result_dir, f"{model_name.lower()}_predictions.csv")
    if os.path.exists(result_file):
        df_pred = pd.read_csv(result_file)
    else:
        df_pred = pd.DataFrame(columns=["Date", "Prediction"])

    df_pred = df_pred[df_pred["Date"] != prediction_date]
    df_pred = pd.concat(
        [df_pred, pd.DataFrame([{"Date": prediction_date, "Prediction": predicted_value}])],
        ignore_index=True
    )
    df_pred.sort_values("Date", inplace=True)
    df_pred.to_csv(result_file, index=False)

    return result_file

def update_actuals(ticker: str, model_name: str):
    result_file = os.path.join("results", ticker, f"{model_name.lower()}_predictions.csv")
    if not os.path.exists(result_file):
        print("❌ Prediction file not found.")
        return

    df = pd.read_csv(result_file)
    if "Actual" not in df.columns:
        df["Actual"] = np.nan

    raw_path = os.path.join("data", "raw", f"{ticker}_raw.csv")
    if not os.path.exists(raw_path):
        print("❌ Local raw data file not found.")
        return

    raw_df = pd.read_csv(raw_path)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    raw_df = raw_df.set_index("Date")

    print(f"📥 Updating actual closing prices for {ticker} from local data...")
    updated = 0

    for i, row in df.iterrows():
        date = pd.to_datetime(row["Date"])
        if not pd.isna(row["Actual"]):
            continue

        try:
            actual_price = raw_df.loc[date, "Close"]
            df.at[i, "Actual"] = actual_price
            updated += 1
            print(f"✅ {date.date()}: {actual_price}")
        except KeyError:
            print(f"⚠️ No data for {date.date()} in raw CSV.")

    df["Error"] = df.apply(
        lambda row: row["Prediction"] - row["Actual"]
        if pd.notna(row["Prediction"]) and pd.notna(row["Actual"]) else np.nan,
        axis=1
    )
    df.to_csv(result_file, index=False)
    print(f"💾 File updated: {result_file}")
    if updated == 0:
        print("ℹ️ No actual values were updated. Check if raw data contains matching dates.")
