import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

TICKER = os.environ.get("TICKER", "AAPL")
DATA_DIR = os.path.join("data", "processed", TICKER)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Załaduj dane
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# Parametry wejściowe
input_shape = X_train.shape[1:]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
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
    x = transformer_encoder(inputs, head_size=64, num_heads=2, ff_dim=64, dropout=0.1)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1)(x)
    return models.Model(inputs, outputs)

# Budowa i kompilacja modelu
model = build_model(input_shape)
model.compile(loss="mse", optimizer="adam")

# Trening
print("\n🚀 Start treningu Transformera...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Ewaluacja
mse = model.evaluate(X_test, y_test)
print(f"\n📉 Test MSE: {mse}")

# Zapis modelu (nowoczesny format .keras)
MODEL_PATH = os.path.join(MODEL_DIR, f"{TICKER}_transformer_model.keras")
model.save(MODEL_PATH)
print(f"💾 Model zapisany do {MODEL_PATH}")
