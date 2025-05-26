import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from tensorflow.keras.models import load_model

# Load ticker from environment variable or use default
TICKER = os.environ.get("TICKER", "AAPL")
PROCESSED_DIR = os.path.join("data", "processed", TICKER)
MODEL_DIR = "models"

# Load test data
X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

results = {}

# === LSTM ===
try:
    lstm_model = load_model(os.path.join(MODEL_DIR, f"{TICKER}_lstm_model.keras"))
    y_pred = lstm_model.predict(X_test)
    results['LSTM'] = mean_squared_error(y_test, y_pred)
except Exception as e:
    results['LSTM'] = None
    print(f"⚠️ LSTM error: {e}")

# === CNN ===
try:
    cnn_model = load_model(os.path.join(MODEL_DIR, f"{TICKER}_cnn_model.keras"))
    y_pred = cnn_model.predict(X_test)
    results['CNN'] = mean_squared_error(y_test, y_pred)
except Exception as e:
    results['CNN'] = None
    print(f"⚠️ CNN error: {e}")

# === Transformer ===
try:
    transformer_model = load_model(os.path.join(MODEL_DIR, f"{TICKER}_transformer_model.keras"))
    y_pred = transformer_model.predict(X_test)
    results['Transformer'] = mean_squared_error(y_test, y_pred)
except Exception as e:
    results['Transformer'] = None
    print(f"⚠️ Transformer error: {e}")

# === XGBoost ===
try:
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(MODEL_DIR, f"{TICKER}_xgboost_model.json"))
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_pred = xgb_model.predict(X_test_flat)
    results['XGBoost'] = mean_squared_error(y_test, y_pred)
except Exception as e:
    results['XGBoost'] = None
    print(f"⚠️ XGBoost error: {e}")

# === ARIMA ===
try:
    arima_path = os.path.join(MODEL_DIR, f"{TICKER}_arima_model.csv")
    if os.path.exists(arima_path):
        arima_df = pd.read_csv(arima_path)
        results['ARIMA'] = mean_squared_error(arima_df['actual'], arima_df['predicted'])
    else:
        results['ARIMA'] = None
        print("⚠️ ARIMA predictions file not found.")
except Exception as e:
    results['ARIMA'] = None
    print(f"⚠️ ARIMA error: {e}")

# === Display results ===
print("\n📊 Model Evaluation Results (MSE):")
for model, mse in results.items():
    if mse is not None:
        print(f"{model:12s} → MSE: {mse:.6f}")
    else:
        print(f"{model:12s} → no data available")
