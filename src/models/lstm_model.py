import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

TICKER = os.environ.get("TICKER", "AAPL")
DATA_DIR = os.path.join("data", "processed", TICKER)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
print("\n🚀 Starting LSTM training...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print(f"\n📉 Test MSE: {mse}")

# Save the model
MODEL_PATH = os.path.join(MODEL_DIR, f"{TICKER}_lstm_model.h5")
model.save(MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")
