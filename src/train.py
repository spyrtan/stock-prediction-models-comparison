import subprocess
import sys
from pathlib import Path

# Path to the src folder
BASE_DIR = Path(__file__).resolve().parent

# List of available training scripts (inside src/models/)
scripts = [
    ("LSTM", BASE_DIR / "models" / "lstm_model.py"),
    ("CNN", BASE_DIR / "models" / "cnn_model.py"),
    ("ARIMA", BASE_DIR / "models" / "arima_model.py"),
    ("Transformer", BASE_DIR / "models" / "transformer.py"),
    ("XGBoost", BASE_DIR / "models" / "xgboost_model.py")
]

print("\n🚀 Starting model training:\n")

for name, script_path in scripts:
    print(f"\n================ {name} =================")
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Error while running {script_path.name}")

print("\n✅ All models have been trained (unless an error occurred).")
