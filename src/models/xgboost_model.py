import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

import numpy as np
import xgboost as xgb
import os
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error
import joblib  # ⬅️ to save scaler

# Load environment variables
TICKER = os.environ.get("TICKER", "AAPL")
SUFFIX = os.environ.get("MODEL_TEMP_SUFFIX", "")

# Use absolute paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / TICKER
MODEL_DIR = BASE_DIR / "models"
TEMP_DIR = MODEL_DIR / "temp"
SCALER_PATH = MODEL_DIR / f"{TICKER}_scaler.save"

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Load preprocessed data
X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

# Flatten input (XGBoost requires 2D input)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Train XGBoost model
print("\n🚀 Starting XGBoost training...")
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train_flat, y_train)

# Predict and evaluate
predictions = model.predict(X_test_flat)
mse = mean_squared_error(y_test, predictions)
print(f"\n📉 Test MSE: {mse:.6f}")

# Determine model save path
model_filename = f"{TICKER}_xgboost_model{SUFFIX}.json"
model_path = TEMP_DIR / model_filename if SUFFIX else MODEL_DIR / model_filename

try:
    model.save_model(model_path)
    print(f"💾 Model saved to {model_path}")
except Exception as e:
    print(f"❌ Failed to save model: {e}")

# Save MSE and model path to JSON
mse_output_path = TEMP_DIR / "xgboost_mse.json"
try:
    with open(mse_output_path, "w") as f:
        json.dump({"mse": mse, "model_path": str(model_path)}, f)
    print(f"📄 MSE and model path written to {mse_output_path}")
except Exception as e:
    print(f"❌ Failed to write JSON: {e}")

# OPTIONAL: Save scaler passed from train.py
scaler_var = os.environ.get("SCALER_OBJ")
if scaler_var and os.path.exists(scaler_var):
    try:
        scaler = joblib.load(scaler_var)
        joblib.dump(scaler, SCALER_PATH)
        print(f"📦 Scaler saved to {SCALER_PATH}")
    except Exception as e:
        print(f"❌ Failed to save scaler: {e}")
