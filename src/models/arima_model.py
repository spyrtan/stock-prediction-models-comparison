import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import os
from sklearn.metrics import mean_squared_error

TICKER = os.environ.get("TICKER", "AAPL")
RAW_PATH = os.path.join("data", "raw", f"{TICKER}_raw.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Załaduj dane
print("\n📥 Ładowanie danych...")
df = pd.read_csv(RAW_PATH)
series = pd.to_numeric(df['Close'], errors='coerce').dropna()

# Podział na dane treningowe i testowe (80/20)
split_idx = int(len(series) * 0.8)
train, test = series[:split_idx], series[split_idx:]

# Trenowanie modelu ARIMA
print("\n🚀 Start treningu ARIMA...")
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

# Predykcja
print("\n🔮 Predykcja na danych testowych...")
predictions = model_fit.forecast(steps=len(test))

# Ewaluacja
mse = mean_squared_error(test, predictions)
print(f"\n📉 Test MSE: {mse}")

# Zapisz predykcje
pred_path = os.path.join(MODEL_DIR, f"{TICKER}_arima_predictions.csv")
pd.DataFrame({"actual": test.values, "predicted": predictions}).to_csv(pred_path, index=False)
print(f"💾 Predykcje zapisane do {pred_path}")
