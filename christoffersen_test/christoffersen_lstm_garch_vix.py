# Re-import necessary libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox

# Setup output directory
output_dir = "christoffersen_test"
os.makedirs(output_dir, exist_ok=True)

# File path
data_path = os.path.join("data", "results_lstm_garch_vix_intraday.csv")

# Load and process data
df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(
    df["test_date"] if "test_date" in df.columns else df["Date"]
)
df.dropna(subset=["actual", "prediction"], inplace=True)

# 95% interval bounds
df["lower"] = -1.96 * df["prediction"]
df["upper"] = 1.96 * df["prediction"]

# Hit sequence (1 if actual is within the interval)
df["hit"] = ((df["actual"] >= df["lower"]) & (df["actual"] <= df["upper"])).astype(int)

# Count runs
hit_sequence = df["hit"].values
runs = []
current = hit_sequence[0]
length = 1
for i in range(1, len(hit_sequence)):
    if hit_sequence[i] == current:
        length += 1
    else:
        runs.append((current, length))
        current = hit_sequence[i]
        length = 1
runs.append((current, length))

# Ljung-Box test for independence
ljung_box = acorr_ljungbox(hit_sequence, lags=[10], return_df=True)

# Plot the hit sequence
plt.figure(figsize=(14, 2.5))
plt.plot(df["Date"], hit_sequence, drawstyle="steps-post", color="darkgreen", lw=2)
plt.yticks([0, 1], labels=["Miss", "Hit"])
plt.title("Hit Sequence on LSTM-GARCH-VIX Forecasts (95% Interval)")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(output_dir, "lstm_garch_vix_hit_sequence.png")
plt.savefig(plot_path)
plt.close()

# Summary statistics
run_lengths = [l for _, l in runs]
summary = {
    "Total Observations": len(df),
    "Total Runs": len(runs),
    "Avg Run Length": np.mean(run_lengths),
    "Max Run Length": np.max(run_lengths),
    "Min Run Length": np.min(run_lengths),
    "Ljung-Box p-value (lag=10)": ljung_box["lb_pvalue"].values[0],
    "Plot Saved": plot_path,
}

# Save to CSV and Markdown table
summary_df = pd.DataFrame([summary])
csv_path = os.path.join(output_dir, "lstm_garch_vix_hit_summary.csv")
txt_path = os.path.join(output_dir, "lstm_garch_vix_hit_summary.txt")
summary_df.to_csv(csv_path, index=False)

with open(txt_path, "w") as f:
    f.write("| Metric                        | Value                  |\n")
    f.write("|------------------------------|------------------------|\n")
    for key, value in summary.items():
        f.write(f"| {key} | {value} |\n")
