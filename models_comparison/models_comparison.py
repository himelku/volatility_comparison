import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ------------------ Define Directories ------------------ #
data_dir = os.path.join("data")
os.makedirs(data_dir, exist_ok=True)

# ------------------ Load GARCH Results ------------------ #
garch_path = os.path.join(data_dir, "results_garch_intraday.csv")
garch = pd.read_csv(garch_path)

# Handle datetime
if "Date" in garch.columns:
    garch["Date"] = pd.to_datetime(garch["Date"])
elif "timestamp" in garch.columns:
    garch["Date"] = pd.to_datetime(garch["timestamp"])
elif "test_date" in garch.columns:
    garch = garch.rename(columns={"test_date": "Date"})
    garch["Date"] = pd.to_datetime(garch["Date"])
else:
    print("❌ GARCH file missing 'Date', 'timestamp', or 'test_date'.")
    garch = pd.DataFrame()

# Filter and print metrics
if not garch.empty and {"actual", "prediction"}.issubset(garch.columns):
    garch = garch[garch["Date"] >= "2015-02-13"]
    mae = mean_absolute_error(garch["actual"], garch["prediction"])
    rmse = np.sqrt(mean_squared_error(garch["actual"], garch["prediction"]))
    print(f"DataFrame: garch | MAE: {mae:.6f} | RMSE: {rmse:.6f}")
else:
    print("⚠️ GARCH data is incomplete or missing required columns.")

# ------------------ Load LSTM-related Results ------------------ #
lstm_files = {
    "lstm": "results_lstm_intraday.csv",
    "lstm_garch": "results_lstm_garch_intraday.csv",
    "lstm_garch_vix": "results_lstm_garch_vix_intraday.csv",
    "lstm_garch_vix_pct_change": "results_lstm_garch_vix_pct_change.csv",
    "lstm_garch_vix_1_layer": "results_lstm_garch_vix_layer_1_intraday.csv",
    "lstm_garch_vix_3_layers": "results_lstm_garch_vix_layer_3_intraday.csv",
    "lstm_garch_vix_lookback_5": "results_lstm_garch_vix_lookback_5.csv",
    "lstm_garch_vix_lookback_66": "results_lstm_garch_vix_lookback_66.csv",
    "lstm_garch_vix_mae_loss": "results_lstm_garch_vix_mae_loss_intraday.csv",
    "lstm_garch_vix_relu": "results_lstm_garch_vix_relu.csv",
}

lstm_dfs = {}
for key, filename in lstm_files.items():
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"⚠️ File not found: {filename}")
        continue

    df = pd.read_csv(path)

    # Check required columns
    if not {"actual", "prediction"}.issubset(df.columns):
        print(f"⚠️ Skipping {filename} — Missing 'actual' or 'prediction'")
        continue

    # Handle datetime column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    elif "timestamp" in df.columns:
        df["Date"] = pd.to_datetime(df["timestamp"])
    elif "test_date" in df.columns:
        df = df.rename(columns={"test_date": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        print(f"⚠️ Skipping {filename} — No 'Date', 'timestamp', or 'test_date'")
        continue

    df.dropna(subset=["actual", "prediction"], inplace=True)
    lstm_dfs[key] = df

    mae = mean_absolute_error(df["actual"], df["prediction"])
    rmse = np.sqrt(mean_squared_error(df["actual"], df["prediction"]))
    print(f"DataFrame: {key} | MAE: {mae:.6f} | RMSE: {rmse:.6f}")

# ------------------ Plotting Function ------------------ #
def plot_data(dataframes, labels, colors, linestyles, title, x_label, y_label):
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Create folder if needed
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Create figure
    plt.figure(figsize=(12, 6))
    for df, label, color, linestyle in zip(dataframes, labels, colors, linestyles):
        plt.plot(df["Date"], df[df.columns[1]], label=label, color=color, linestyle=linestyle)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    filename = f"{title.replace(' ', '_').replace('.', '').lower()}.png"
    filepath = plots_dir / filename
    plt.savefig(filepath)
    print(f"✅ Plot saved: {filepath}")
    plt.close()

# ------------------ Plotting Section ------------------ #

# GARCH Plot
if not garch.empty:
    plot_data(
        [garch[["Date", "prediction"]], garch[["Date", "actual"]]],
        ["Predicted Volatility", "Actual"],
        ["royalblue", "red"],
        ["-", "--"],
        "GARCH Predictions vs. Actual",
        "Date",
        "Volatility",
    )

# Generic helper for plotting any valid models
def plot_if_exists(keys, labels, colors, linestyles, title):
    dataframes = []
    for k in keys:
        if k in lstm_dfs:
            label = labels[len(dataframes)]
            df = lstm_dfs[k]
            if "actual" in label.lower():
                dataframes.append(df[["Date", "actual"]])
            else:
                dataframes.append(df[["Date", "prediction"]])
    if dataframes:
        plot_data(dataframes, labels, colors, linestyles, title, "Date", "Volatility")

# Plot key comparisons
plot_if_exists(["lstm", "lstm"], ["LSTM", "Actual"], ["gold", "red"], ["-", "--"], "LSTM vs. Actual")
plot_if_exists(["lstm_garch", "lstm_garch"], ["LSTM-GARCH", "Actual"], ["deepskyblue", "red"], ["-", "--"], "LSTM-GARCH vs. Actual")
plot_if_exists(["lstm_garch_vix", "lstm_garch_vix"], ["LSTM-GARCH-VIX", "Actual"], ["springgreen", "red"], ["-", "--"], "LSTM-GARCH-VIX vs. Actual")

# Additional comparisons
plot_if_exists(["lstm_garch_vix_mae_loss", "lstm_garch_vix", "lstm_garch_vix"], ["MAE Loss", "MSE Loss", "Actual"], ["purple", "springgreen", "red"], ["-", "-", "--"], "MAE vs. MSE Loss")
plot_if_exists(["lstm_garch_vix", "lstm_garch_vix_pct_change", "lstm_garch_vix"], ["Log Return", "Pct Change", "Actual"], ["springgreen", "gold", "red"], ["-", "-", "--"], "Log Return vs. Pct Change")
plot_if_exists(["lstm_garch_vix_lookback_66", "lstm_garch_vix", "lstm_garch_vix_lookback_5", "lstm_garch_vix"], ["Lookback 66", "Lookback 22", "Lookback 5", "Actual"], ["purple", "springgreen", "gold", "red"], ["-", "-", "-", "--"], "Lookback Comparison")
plot_if_exists(["lstm_garch_vix_3_layers", "lstm_garch_vix", "lstm_garch_vix_1_layer", "lstm_garch_vix"], ["3 Layers", "2 Layers", "1 Layer", "Actual"], ["purple", "springgreen", "gold", "red"], ["-", "-", "-", "--"], "LSTM Layer Depth Comparison")
plot_if_exists(["lstm_garch_vix_relu", "lstm_garch_vix", "lstm_garch_vix"], ["ReLU", "Tanh", "Actual"], ["gold", "springgreen", "red"], ["-", "-", "--"], "Activation Function Comparison")

# ------------------ EWMA Evaluation ------------------ #
ewma_path = os.path.join(data_dir, "results_ewma.csv")
if os.path.exists(ewma_path):
    ewma = pd.read_csv(ewma_path)
    ewma["Date"] = pd.to_datetime(ewma["Date"])
    if "ewma_volatility" in ewma.columns and "log_returns" in ewma.columns:
        actual = ewma["log_returns"].rolling(26).std().dropna()
        predicted = ewma["ewma_volatility"][25:]
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        print(f"DataFrame: ewma | MAE: {mae:.6f} | RMSE: {rmse:.6f}")
    else:
        print("⚠️ EWMA file missing required columns.")
else:
    print("⚠️ results_ewma.csv not found.")

# ------------------ EWMA Plot ------------------ #
if 'ewma' in locals():

    # Compute rolling std only on numeric column
    rolling_std = ewma[["log_returns"]].rolling(26).std().rename(columns={"log_returns": "rolling_std"})

    # Combine with EWMA volatility (make sure 'Date' is preserved)
    combined_ewma = pd.concat([ewma[["Date", "ewma_volatility"]], rolling_std], axis=1).dropna()

    # Plotting
    plot_data(
        [
            combined_ewma[["Date", "ewma_volatility"]],
            combined_ewma[["Date", "rolling_std"]]
        ],
        labels=["EWMA Volatility", "Rolling Std Dev"],
        colors=["blue", "gray"],
        linestyles=["-", "--"],
        title="EWMA vs Rolling Std Dev (26 periods)",
        x_label="Date",
        y_label="Volatility"
    )
