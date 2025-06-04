import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path

TICKER = os.environ.get("TICKER", "AAPL")
PROCESSED_DIR = os.path.join("data", "processed", TICKER)
MODEL_DIR = "models"

X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

scaler_path = Path(MODEL_DIR) / f"{TICKER}_scaler.save"
if not scaler_path.exists():
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
scaler = joblib.load(scaler_path)

def inverse_close_only(values):
    padded = np.hstack([values.reshape(-1, 1), np.zeros((len(values), 4))])
    return scaler.inverse_transform(padded)[:, 0]

y_test_inv = inverse_close_only(y_test)
results = {}

# === LSTM ===
try:
    lstm_model = load_model(os.path.join(MODEL_DIR, f"{TICKER}_lstm_model.keras"))
    y_pred = lstm_model.predict(X_test).flatten()
    y_pred_inv = inverse_close_only(y_pred)
    results['LSTM'] = {
        "MSE_scaled": mean_squared_error(y_test, y_pred),
        "MSE_real": mean_squared_error(y_test_inv, y_pred_inv),
        "MAE": mean_absolute_error(y_test_inv, y_pred_inv),
        "R2": r2_score(y_test_inv, y_pred_inv),
        "MAPE": np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    }
except Exception as e:
    results['LSTM'] = None
    print(f"⚠️ LSTM error: {e}")

# === CNN ===
try:
    cnn_model = load_model(os.path.join(MODEL_DIR, f"{TICKER}_cnn_model.keras"))
    y_pred = cnn_model.predict(X_test).flatten()
    y_pred_inv = inverse_close_only(y_pred)
    results['CNN'] = {
        "MSE_scaled": mean_squared_error(y_test, y_pred),
        "MSE_real": mean_squared_error(y_test_inv, y_pred_inv),
        "MAE": mean_absolute_error(y_test_inv, y_pred_inv),
        "R2": r2_score(y_test_inv, y_pred_inv),
        "MAPE": np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    }
except Exception as e:
    results['CNN'] = None
    print(f"⚠️ CNN error: {e}")

# === Transformer ===
try:
    transformer_model = load_model(os.path.join(MODEL_DIR, f"{TICKER}_transformer_model.keras"))
    y_pred = transformer_model.predict(X_test).flatten()
    y_pred_inv = inverse_close_only(y_pred)
    results['Transformer'] = {
        "MSE_scaled": mean_squared_error(y_test, y_pred),
        "MSE_real": mean_squared_error(y_test_inv, y_pred_inv),
        "MAE": mean_absolute_error(y_test_inv, y_pred_inv),
        "R2": r2_score(y_test_inv, y_pred_inv),
        "MAPE": np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    }
except Exception as e:
    results['Transformer'] = None
    print(f"⚠️ Transformer error: {e}")

# === XGBoost ===
try:
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(MODEL_DIR, f"{TICKER}_xgboost_model.json"))
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_pred = xgb_model.predict(X_test_flat)
    y_pred_inv = inverse_close_only(y_pred)
    results['XGBoost'] = {
        "MSE_scaled": mean_squared_error(y_test, y_pred),
        "MSE_real": mean_squared_error(y_test_inv, y_pred_inv),
        "MAE": mean_absolute_error(y_test_inv, y_pred_inv),
        "R2": r2_score(y_test_inv, y_pred_inv),
        "MAPE": np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    }
except Exception as e:
    results['XGBoost'] = None
    print(f"⚠️ XGBoost error: {e}")

# === ARIMA ===
try:
    arima_path = os.path.join(MODEL_DIR, f"{TICKER}_arima_predictions.csv")
    if not os.path.exists(arima_path):
        arima_path = os.path.join(MODEL_DIR, "temp", f"{TICKER}_arima_predictions__temp_run_1.csv")

    if os.path.exists(arima_path):
        arima_df = pd.read_csv(arima_path)
        actual = arima_df["actual"].values
        predicted = arima_df["predicted"].values
        results['ARIMA'] = {
            "MSE": mean_squared_error(actual, predicted),
            "MAE": mean_absolute_error(actual, predicted),
            "R2": r2_score(actual, predicted),
            "MAPE": np.mean(np.abs((actual - predicted) / actual)) * 100
        }
    else:
        results['ARIMA'] = None
        print("⚠️ ARIMA prediction CSV not found.")
except Exception as e:
    results['ARIMA'] = None
    print(f"⚠️ ARIMA error: {e}")

# === Display results ===
print("\n📊 Model Evaluation Results:")
for model, metrics in results.items():
    if metrics is not None:
        print(f"\n🔍 {model}")
        for key, value in metrics.items():
            print(f"   {key}: {value:.6f}")
    else:
        print(f"\n🔍 {model}: no data available")
