import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import os
import json
from pathlib import Path

# Load environment variables
TICKER = os.environ.get("TICKER", "AAPL")
SUFFIX = os.environ.get("MODEL_TEMP_SUFFIX", "")

# Use absolute paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / TICKER
MODEL_DIR = BASE_DIR / "models"
TEMP_DIR = MODEL_DIR / "temp"

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Load preprocessed data
X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

# Build the CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
print("\n🚀 Starting CNN training...")
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print(f"\n📉 Test MSE: {mse:.6f}")

# Save model
model_filename = f"{TICKER}_cnn_model{SUFFIX}.keras"
model_path = TEMP_DIR / model_filename if SUFFIX else MODEL_DIR / model_filename

try:
    model.save(model_path)
    print(f"💾 Model saved to {model_path}")
except Exception as e:
    print(f"❌ Failed to save model: {e}")

# Save MSE to JSON
mse_output_path = TEMP_DIR / "cnn_mse.json"
try:
    with open(mse_output_path, "w") as f:
        json.dump({"mse": mse, "model_path": str(model_path)}, f)
    print(f"📄 MSE and model path written to {mse_output_path}")
except Exception as e:
    print(f"❌ Failed to write JSON: {e}")
