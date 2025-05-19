import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src import preprocess
from src import save_data

TICKER = os.environ.get("TICKER", "AAPL")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Data parameters
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

# Extract Close as a time series
series = df["Close"].dropna()
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

print("\n🚀 Starting ARIMA training...")
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

print("\n🔮 Forecasting on test data...")
predictions = model_fit.forecast(steps=len(test))

# Evaluation
mse = mean_squared_error(test, predictions)
print(f"\n📉 Test MSE: {mse}")

# Save predictions
pred_path = os.path.join(MODEL_DIR, f"{TICKER}_arima_predictions.csv")
df_result = pd.DataFrame({
    "actual": test.values.squeeze(),
    "predicted": predictions.squeeze()
})
df_result.to_csv(pred_path, index=False)
print(f"💾 Predictions saved to {pred_path}")
