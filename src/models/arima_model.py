import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Include project root in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Load environment variables
TICKER = os.environ.get("TICKER", "AAPL")
SUFFIX = os.environ.get("MODEL_TEMP_SUFFIX", "")

# Absolute paths
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
TEMP_DIR = MODEL_DIR / "temp"
RAW_PATH = BASE_DIR / "data" / "raw" / f"{TICKER}_raw.csv"

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Load raw data from CSV
print(f"\n📂 Loading data from {RAW_PATH}...")
df = pd.read_csv(RAW_PATH, parse_dates=["Date"])
df = df.sort_values("Date").dropna()

# Extract and log-transform the closing prices
series = np.log(df["Close"].values)
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# Train ARIMA model
print("\n🚀 Starting ARIMA training...")
model = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore', trace=False)

# Forecast
print("\n🔮 Forecasting on test data...")
predictions_log = model.predict(n_periods=len(test))
predictions = np.exp(predictions_log)
test_exp = np.exp(test)

# Evaluate
mse = mean_squared_error(test_exp, predictions)
mae = mean_absolute_error(test_exp, predictions)
r2 = r2_score(test_exp, predictions)
mape = np.mean(np.abs((test_exp - predictions) / test_exp)) * 100

print(f"\n📉 Test MSE: {mse:.6f}")

# Save predictions to CSV
df_result = pd.DataFrame({
    "actual": test_exp,
    "predicted": predictions
})

file_name = f"{TICKER}_arima_predictions{SUFFIX}.csv"
pred_path = TEMP_DIR / file_name if SUFFIX else MODEL_DIR / file_name

df_result.to_csv(pred_path, index=False)
print(f"💾 Predictions saved to {pred_path}")

# Save model
model_filename = f"{TICKER}_arima_model{SUFFIX}.pkl"
model_path = TEMP_DIR / model_filename if SUFFIX else MODEL_DIR / model_filename
joblib.dump(model, model_path)
print(f"💾 Model saved to {model_path}")

# Save metrics and paths to JSON
mse_output_path = TEMP_DIR / "arima_mse.json"
with open(mse_output_path, "w") as f:
    json.dump({
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "model_path": str(pred_path),
        "model_file": str(model_path)
    }, f)
print(f"📄 Metrics written to {mse_output_path}")
