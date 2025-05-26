import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Include project root in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src import preprocess
from src import save_data

# Load environment variables
TICKER = os.environ.get("TICKER", "AAPL")
SUFFIX = os.environ.get("MODEL_TEMP_SUFFIX", "")

# Absolute paths
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
TEMP_DIR = MODEL_DIR / "temp"

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Data preparation parameters
START = "2015-01-01"
END = "2024-12-31"
INTERVAL = "1d"
WINDOW_SIZE = 30

# Fetch and preprocess data
X_train, y_train, X_test, y_test, scaler, df = preprocess.prepare_data(
    ticker=TICKER,
    start=START,
    end=END,
    interval=INTERVAL,
    window_size=WINDOW_SIZE
)

# Extract time series from Close column
series = df["Close"].dropna()
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# Train ARIMA model
print("\n🚀 Starting ARIMA training...")
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Forecast
print("\n🔮 Forecasting on test data...")
predictions = model_fit.forecast(steps=len(test))

# Evaluate
mse = mean_squared_error(test, predictions)
print(f"\n📉 Test MSE: {mse:.6f}")

# Determine save path
file_name = f"{TICKER}_arima_predictions{SUFFIX}.csv"
pred_path = TEMP_DIR / file_name if SUFFIX else MODEL_DIR / file_name

# Save predictions to CSV
try:
    df_result = pd.DataFrame({
        "actual": test.values.squeeze(),
        "predicted": predictions.squeeze()
    })
    df_result.to_csv(pred_path, index=False)
    print(f"💾 Predictions saved to {pred_path}")
except Exception as e:
    print(f"❌ Failed to save predictions CSV: {e}")

# Save MSE and prediction path to JSON
mse_output_path = TEMP_DIR / "arima_mse.json"
try:
    with open(mse_output_path, "w") as f:
        json.dump({"mse": mse, "model_path": str(pred_path)}, f)
    print(f"📄 MSE and model path written to {mse_output_path}")
except Exception as e:
    print(f"❌ Failed to write JSON: {e}")
