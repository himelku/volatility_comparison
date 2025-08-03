# Re-import necessary libraries after kernel reset
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox

# Ensure plot directory exists
output_dir = "christoffersen_test"
os.makedirs(output_dir, exist_ok=True)

# Reload the uploaded file
data_path = os.path.join("data", "results_garch_intraday.csv")

# Read and process
real_garch_df = pd.read_csv(data_path)
real_garch_df["Date"] = pd.to_datetime(real_garch_df["Date"])
real_garch_df.dropna(subset=["actual", "prediction"], inplace=True)

# Create 95% prediction interval (±1.96 * predicted volatility)
real_garch_df["lower"] = -1.96 * real_garch_df["prediction"]
real_garch_df["upper"] = 1.96 * real_garch_df["prediction"]

# Determine if actual value is within interval
real_garch_df["hit"] = (
    (real_garch_df["actual"] >= real_garch_df["lower"])
    & (real_garch_df["actual"] <= real_garch_df["upper"])
).astype(int)

# Count runs
hit_sequence = real_garch_df["hit"].values
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

# Ljung-Box test
ljung_real = acorr_ljungbox(hit_sequence, lags=[10], return_df=True)

# Plot hit sequence
plt.figure(figsize=(14, 2.5))
plt.plot(
    real_garch_df["Date"], hit_sequence, drawstyle="steps-post", color="darkblue", lw=2
)
plt.yticks([0, 1], labels=["Miss", "Hit"])
plt.title("Hit Sequence on Real GARCH Forecasts (95% Interval)")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(output_dir, "garch_hit_sequence.png")
plt.savefig(plot_path)
plt.close()

# Create summary
run_lengths = [l for v, l in runs]
summary_real = {
    "Total Observations": len(real_garch_df),
    "Total Runs": len(runs),
    "Avg Run Length": np.mean(run_lengths),
    "Max Run Length": np.max(run_lengths),
    "Min Run Length": np.min(run_lengths),
    "Ljung-Box p-value (lag=10)": ljung_real["lb_pvalue"].values[0],
    "Plot Saved": plot_path,
}

summary_real

# Save the GARCH summary dictionary to a CSV and a Markdown-friendly text file
summary_df = pd.DataFrame([summary_real])

# Define paths
csv_path = "christoffersen_test/garch_hit_summary.csv"
txt_path = "christoffersen_test/garch_hit_summary.txt"

# Save as CSV
summary_df.to_csv(csv_path, index=False)

# Save as Markdown-style table
with open(txt_path, "w") as f:
    f.write("| Metric                        | Value                  |\n")
    f.write("|------------------------------|------------------------|\n")
    for key, value in summary_real.items():
        f.write(f"| {key} | {value} |\n")

csv_path, txt_path
