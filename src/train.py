import subprocess
import sys
from pathlib import Path

# Ścieżka do folderu src
BASE_DIR = Path(__file__).resolve().parent

# Lista dostępnych skryptów do trenowania (w src/models/)
scripts = [
    ("LSTM", BASE_DIR / "models" / "lstm_model.py"),
    ("CNN", BASE_DIR / "models" / "cnn_model.py"),
    ("ARIMA", BASE_DIR / "models" / "arima_model.py"),
    ("Transformer", BASE_DIR / "models" / "transformer.py"),
    ("XGBoost", BASE_DIR / "models" / "xgboost_model.py")
]

print("\n🚀 Uruchamianie treningów modeli:\n")

for name, script_path in scripts:
    print(f"\n================ {name} =================")
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Wystąpił błąd podczas uruchamiania {script_path.name}")

print("\n✅ Wszystkie modele zostały przetrenowane (o ile nie było błędów).")
