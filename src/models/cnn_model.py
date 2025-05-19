import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
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

# Build CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
print("\n🚀 Starting CNN training...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print(f"\n📉 Test MSE: {mse}")

# Save the model
MODEL_PATH = os.path.join(MODEL_DIR, f"{TICKER}_cnn_model.h5")
model.save(MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")
