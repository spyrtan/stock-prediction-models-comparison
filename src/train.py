import subprocess
import sys
from pathlib import Path
import os
import shutil
import json

# Base project directory
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "src" / "models"
TEMP_DIR = BASE_DIR / "models" / "temp"
SCALER_PATH = BASE_DIR / "models" / f"{os.environ.get('TICKER', 'MODEL')}_scaler.save"

# Map of available models and their script paths
all_scripts = {
    "LSTM": MODEL_DIR / "lstm_model.py",
    "CNN": MODEL_DIR / "cnn_model.py",
    "ARIMA": MODEL_DIR / "arima_model.py",
    "Transformer": MODEL_DIR / "transformer.py",
    "XGBoost": MODEL_DIR / "xgboost_model.py"
}

# Determine selected models from environment variable, if any
selected_env = os.environ.get("SELECTED_MODELS")
if selected_env:
    selected_models = [name.strip() for name in selected_env.split(",") if name.strip() in all_scripts]
    scripts = [(name, all_scripts[name]) for name in selected_models]
else:
    scripts = list(all_scripts.items())

# Get number of training repetitions from environment
try:
    num_repeats = int(os.environ.get("TRAIN_REPEAT", "1"))
    if num_repeats < 1:
        raise ValueError
except ValueError:
    print("⚠️ Invalid TRAIN_REPEAT value. Using default (1).")
    num_repeats = 1

# Ensure temp folder for intermediate results
os.makedirs(TEMP_DIR, exist_ok=True)

print(f"\n🚀 Starting training ({num_repeats}x per model)...\n")

# Inform about scaler availability
if SCALER_PATH.exists():
    print(f"📦 Using cached scaler from {SCALER_PATH}")
else:
    print("ℹ️ No scaler found yet — will wait for model scripts to create it.")

# Train models
for name, script_path in scripts:
    print(f"\n================ {name} =================")

    best_mse = float("inf")
    best_model_path = None
    best_run_index = None

    for i in range(num_repeats):
        print(f"\n🔄 Training round {i+1}/{num_repeats} for {name}")

        env = os.environ.copy()
        env["MODEL_TEMP_SUFFIX"] = f"__temp_run_{i+1}"
        env["PYTHONPATH"] = str(BASE_DIR)
        env["TEMP_DIR_OVERRIDE"] = str(TEMP_DIR)

        # Pass scaler path if available
        if SCALER_PATH.exists():
            env["SCALER_OBJ"] = str(SCALER_PATH)

        try:
            subprocess.run([sys.executable, str(script_path)], check=True, env=env)
        except subprocess.CalledProcessError:
            print(f"❌ Error while running {script_path.name}")
            continue

        temp_mse_path = TEMP_DIR / f"{name.lower()}_mse.json"
        if not temp_mse_path.exists():
            print(f"⚠️ Skipped: No MSE file generated for {name}")
            continue

        try:
            with open(temp_mse_path, "r") as f:
                result = json.load(f)
                current_mse = float(result["mse"])
                print(f"📉 MSE: {current_mse:.6f}")
        except Exception as e:
            print(f"⚠️ Could not read MSE from file: {temp_mse_path} — {e}")
            continue

        if current_mse < best_mse:
            best_mse = current_mse
            best_model_path = result.get("model_file" if name == "ARIMA" else "model_path")
            best_run_index = i + 1

    if best_model_path:
        final_model_filename = f"{os.environ.get('TICKER', 'MODEL')}_{name.lower()}_model"
        ext = Path(best_model_path).suffix
        final_model_path = BASE_DIR / "models" / f"{final_model_filename}{ext}"
        shutil.copy(best_model_path, final_model_path)
        print(f"✅ Best model for {name} saved to: {final_model_path}")

        if name == "ARIMA" and best_run_index is not None:
            pred_src = TEMP_DIR / f"{os.environ.get('TICKER', 'MODEL')}_arima_predictions__temp_run_{best_run_index}.csv"
            pred_dst = BASE_DIR / "models" / f"{os.environ.get('TICKER', 'MODEL')}_arima_predictions.csv"
            if pred_src.exists():
                shutil.copy(pred_src, pred_dst)
                print(f"📄 ARIMA predictions copied to: {pred_dst}")
    else:
        print(f"⚠️ No valid model was saved for {name}")

# Clean up temp files
shutil.rmtree(TEMP_DIR, ignore_errors=True)

print("\n✅ Training process complete.")
