import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from tensorflow.keras.models import load_model

TICKER = os.environ.get("TICKER", "AAPL")
PROCESSED_DIR = os.path.join("data", "processed", TICKER)
MODEL_DIR = "models"
RAW_PATH = os.path.join("data", "raw", f"{TICKER}_raw.csv")

results = {}

# === LSTM ===
lstm_model = load_model(os.path.join(MODEL_DIR, f"{TICKER}_lstm_model.h5"))
X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
y_pred = lstm_model.predict(X_test)
results['LSTM'] = mean_squared_error(y_test, y_pred)

# === CNN ===
cnn_model = load_model(os.path.join(MODEL_DIR, f"{TICKER}_cnn_model.h5"))
y_pred = cnn_model.predict(X_test)
results['CNN'] = mean_squared_error(y_test, y_pred)

# === Transformer ===
transformer_model = load_model(os.path.join(MODEL_DIR, f"{TICKER}_transformer_model.keras"))
y_pred = transformer_model.predict(X_test)
results['Transformer'] = mean_squared_error(y_test, y_pred)

# === XGBoost ===
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(os.path.join(MODEL_DIR, f"{TICKER}_xgboost_model.json"))
X_test_flat = X_test.reshape(X_test.shape[0], -1)
y_pred = xgb_model.predict(X_test_flat)
results['XGBoost'] = mean_squared_error(y_test, y_pred)

# === ARIMA ===
arima_pred_path = os.path.join(MODEL_DIR, f"{TICKER}_arima_predictions.csv")
if os.path.exists(arima_pred_path):
    arima_df = pd.read_csv(arima_pred_path)
    results['ARIMA'] = mean_squared_error(arima_df['actual'], arima_df['predicted'])
else:
    results['ARIMA'] = None

# === Wyświetlenie wyników ===
print("\n📊 Wyniki ewaluacji (MSE):")
for model, mse in results.items():
    if mse is not None:
        print(f"{model:12s} → MSE: {mse:.6f}")
    else:
        print(f"{model:12s} → brak danych")
