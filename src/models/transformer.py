import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
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

input_shape = X_train.shape[1:]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.2):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return layers.LayerNormalization(epsilon=1e-6)(x + res)

def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_embedding = layers.Embedding(input_dim=1000, output_dim=input_shape[-1])(positions)
    pos_embedding = tf.expand_dims(pos_embedding, axis=0)
    x = layers.Add()([inputs, pos_embedding])

    for _ in range(4):  # 4 Transformer blocks
        x = transformer_encoder(x, head_size=64, num_heads=8, ff_dim=128, dropout=0.2)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)

    return models.Model(inputs, outputs)

model = build_model(input_shape)
model.compile(loss="mse", optimizer="adam")

print("\n🚀 Starting Transformer training...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, batch_size=32)

mse = model.evaluate(X_test, y_test)
print(f"\n📉 Test MSE: {mse:.6f}")

model_filename = f"{TICKER}_transformer_model{SUFFIX}.keras"
model_path = TEMP_DIR / model_filename if SUFFIX else MODEL_DIR / model_filename

try:
    model.save(model_path)
    print(f"💾 Model saved to {model_path}")
except Exception as e:
    print(f"❌ Failed to save model: {e}")

mse_output_path = TEMP_DIR / "transformer_mse.json"
try:
    with open(mse_output_path, "w") as f:
        json.dump({"mse": mse, "model_path": str(model_path)}, f)
    print(f"📄 MSE and model path written to {mse_output_path}")
except Exception as e:
    print(f"❌ Failed to write JSON: {e}")

scaler_var = os.environ.get("SCALER_OBJ")
if scaler_var and os.path.exists(scaler_var):
    try:
        scaler = joblib.load(scaler_var)
        joblib.dump(scaler, SCALER_PATH)
        print(f"📦 Scaler saved to {SCALER_PATH}")
    except Exception as e:
        print(f"❌ Failed to save scaler: {e}")
