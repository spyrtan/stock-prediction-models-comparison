import numpy as np
import xgboost as xgb
import os
from sklearn.metrics import mean_squared_error

TICKER = os.environ.get("TICKER", "AAPL")
DATA_DIR = os.path.join("data", "processed", TICKER)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Załaduj dane
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# Flatten input (XGBoost nie obsługuje danych 3D)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Trenowanie modelu
print("\n🚀 Start treningu XGBoost...")
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train_flat, y_train)

# Predykcja i ewaluacja
preds = model.predict(X_test_flat)
mse = mean_squared_error(y_test, preds)
print(f"\n📉 Test MSE: {mse}")

# Zapis modelu
MODEL_PATH = os.path.join(MODEL_DIR, f"{TICKER}_xgboost_model.json")
model.save_model(MODEL_PATH)
print(f"💾 Model zapisany do {MODEL_PATH}")
