import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import json
from pathlib import Path
import joblib

TICKER = os.environ.get("TICKER", "AAPL")
SUFFIX = os.environ.get("MODEL_TEMP_SUFFIX", "")
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / TICKER
MODEL_DIR = BASE_DIR / "models"
TEMP_DIR = MODEL_DIR / "temp"
SCALER_PATH = MODEL_DIR / f"{TICKER}_scaler.save"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

# Build deeper LSTM model with Dropout
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint_path = TEMP_DIR / f"lstm_checkpoint{SUFFIX}.keras"
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True)

print("\n🚀 Starting LSTM training...")
model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Evaluate
mse = model.evaluate(X_test, y_test)
print(f"\n📉 Test MSE: {mse:.6f}")

# Save final model
model_filename = f"{TICKER}_lstm_model{SUFFIX}.keras"
model_path = TEMP_DIR / model_filename if SUFFIX else MODEL_DIR / model_filename

try:
    model.save(model_path)
    print(f"💾 Model saved to {model_path}")
except Exception as e:
    print(f"❌ Failed to save model: {e}")

# Save MSE to JSON
mse_output_path = TEMP_DIR / "lstm_mse.json"
try:
    with open(mse_output_path, "w") as f:
        json.dump({"mse": mse, "model_path": str(model_path)}, f)
    print(f"📄 MSE and model path written to {mse_output_path}")
except Exception as e:
    print(f"❌ Failed to write JSON: {e}")

# Save scaler (optional)
scaler_var = os.environ.get("SCALER_OBJ")
if scaler_var and os.path.exists(scaler_var):
    try:
        scaler = joblib.load(scaler_var)
        joblib.dump(scaler, SCALER_PATH)
        print(f"📦 Scaler saved to {SCALER_PATH}")
    except Exception as e:
        print(f"❌ Failed to save scaler: {e}")
