# Re-import necessary libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox

# Define file paths
# Reload the uploaded file
ewma_vix_file = os.path.join("data", "results_lstm_garch_ewma_vix_intraday.csv")
pure_lstm_file = os.path.join("data", "results_lstm_intraday.csv")

# Output directory
output_dir = "christoffersen_test"
os.makedirs(output_dir, exist_ok=True)


# Function to compute and save hit sequence stats
def christoffersen_hit_analysis(file_path, label, filename_prefix):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(
        df["test_date"] if "test_date" in df.columns else df["Date"]
    )
    df.dropna(subset=["actual", "prediction"], inplace=True)

    # Create prediction interval and hit column
    df["lower"] = -1.96 * df["prediction"]
    df["upper"] = 1.96 * df["prediction"]
    df["hit"] = ((df["actual"] >= df["lower"]) & (df["actual"] <= df["upper"])).astype(
        int
    )

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

    # Ljung-Box test
    ljung = acorr_ljungbox(hit_sequence, lags=[10], return_df=True)

    # Plot
    plt.figure(figsize=(14, 2.5))
    plt.plot(df["Date"], hit_sequence, drawstyle="steps-post", lw=2, label=label)
    plt.yticks([0, 1], labels=["Miss", "Hit"])
    plt.title(f"Hit Sequence on {label} Forecasts (95% Interval)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{filename_prefix}_hit_sequence.png")
    plt.savefig(plot_path)
    plt.close()

    # Summary
    run_lengths = [l for _, l in runs]
    summary = {
        "Model": label,
        "Total Observations": len(df),
        "Total Runs": len(runs),
        "Avg Run Length": np.mean(run_lengths),
        "Max Run Length": np.max(run_lengths),
        "Min Run Length": np.min(run_lengths),
        "Ljung-Box p-value (lag=10)": ljung["lb_pvalue"].values[0],
        "Plot Saved": plot_path,
    }

    # Export
    summary_df = pd.DataFrame([summary])
    csv_path = os.path.join(output_dir, f"{filename_prefix}_hit_summary.csv")
    txt_path = os.path.join(output_dir, f"{filename_prefix}_hit_summary.txt")
    summary_df.to_csv(csv_path, index=False)

    with open(txt_path, "w") as f:
        f.write("| Metric                        | Value                  |\n")
        f.write("|------------------------------|------------------------|\n")
        for key, value in summary.items():
            f.write(f"| {key} | {value} |\n")

    return summary


# Run analysis for both models
ewma_vix_summary = christoffersen_hit_analysis(
    ewma_vix_file, "LSTM-GARCH-EWMA-VIX", "lstm_garch_ewma_vix"
)
pure_lstm_summary = christoffersen_hit_analysis(
    pure_lstm_file, "Pure LSTM", "pure_lstm"
)

ewma_vix_summary, pure_lstm_summary
