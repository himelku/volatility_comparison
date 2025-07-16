import os
import subprocess
import shutil
import glob
from datetime import datetime

# ------------- Configuration -------------
# Define folders to clean and recreate
output_folders = ["data/results", "models", "reports", "plots"]

# LSTM model script(s) to run
lstm_models_to_run = [
    "lstm_intraday.py",
    # "sequence_length_66.py",
    # "lstm_garch_vix_intraday.py",
    # "lstm_garch_ewma_vix.py"
]
``
# ------------- Utility Functions -------------


def clean_outputs():
    print("\n🧹 Cleaning previous results...")

    for folder in output_folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    # Remove leftover result CSVs and model weights in root or /data/
    for pattern in ["*.h5", "*.csv", "data/*.csv"]:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except:
                pass


def log_and_run(script_name):
    print(f"\n🚀 Running: {script_name}")
    subprocess.run(["python", script_name], check=True)


# ------------- Run the Pipeline -------------

if __name__ == "__main__":
    print("📊 MASTER VOLATILITY PIPELINE STARTED")
    print(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 0: Cleanup
    clean_outputs()

    # Step 1: Preprocess data
    log_and_run("prepare_data_lstm_intraday.py")

    # Step 2: Run GARCH model
    log_and_run("garch.py")

    # Step 3: Train LSTM or hybrid models
    for model_script in lstm_models_to_run:
        log_and_run(model_script)

    # Step 4: Compare all models
    log_and_run("models_comparison.py")

    print("\n✅ FULL PIPELINE COMPLETED SUCCESSFULLY!")
