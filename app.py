import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --------------------------- App Configuration --------------------------- #
st.set_page_config(page_title="Volatility Model Comparison", layout="wide")
st.title("📈 Volatility Prediction Dashboard")

# --------------------------- Utility Functions --------------------------- #
data_dir = Path("data")
plot_dir = Path("plots")

# Available result files mapped to readable names
model_files = {
    "GARCH": "results_garch_intraday.csv",
    "LSTM": "results_lstm_intraday.csv",
    "LSTM-GARCH": "results_lstm_garch_intraday.csv",
    "LSTM-GARCH-VIX": "results_lstm_garch_vix_intraday.csv",
    "LSTM-GARCH-VIX (1 Layer)": "results_lstm_garch_vix_layer_1_intraday.csv",
    "LSTM-GARCH-VIX (3 Layers)": "results_lstm_garch_vix_layer_3_intraday.csv",
    "LSTM-GARCH-VIX (Lookback 5)": "results_lstm_garch_vix_lookback_5.csv",
    "LSTM-GARCH-VIX (Lookback 66)": "results_lstm_garch_vix_lookback_66.csv",
    "LSTM-GARCH-VIX (MAE Loss)": "results_lstm_garch_vix_mae_loss_intraday.csv",
    "LSTM-GARCH-VIX (Pct Change)": "results_lstm_garch_vix_pct_change.csv",
    "LSTM-GARCH-VIX (ReLU)": "results_lstm_garch_vix_relu.csv",
    "LSTM-GARCH-EWMA-VIX": "results_lstm_garch_ewma_vix_intraday.csv",
    "EWMA": "results_ewma.csv",
}


# Load and clean data
def load_model_data(model_key):
    file_path = data_dir / model_files[model_key]
    if not file_path.exists():
        st.warning(f"File not found: {model_files[model_key]}")
        return None

    df = pd.read_csv(file_path)

    # Rename datetime column to 'Date'
    for col in ["Date", "timestamp", "test_date"]:
        if col in df.columns:
            df["Date"] = pd.to_datetime(df[col])
            break

    df.dropna(subset=["Date"], inplace=True)
    return df


# Plot selected models
def plot_models(selected_models):
    fig, ax = plt.subplots(figsize=(14, 6))

    for model in selected_models:
        df = load_model_data(model)
        if df is None:
            continue

        if model == "EWMA":
            if "log_returns" in df.columns and "ewma_volatility" in df.columns:
                rolling_std = df["log_returns"].rolling(26).std().dropna()
                ax.plot(
                    df["Date"].iloc[25:],
                    df["ewma_volatility"].iloc[25:],
                    label="EWMA Volatility",
                    linestyle="-",
                    color="blue",
                )
                ax.plot(
                    df["Date"].iloc[25:],
                    rolling_std,
                    label="Rolling Std Dev",
                    linestyle="--",
                    color="gray",
                )
        else:
            if {"actual", "prediction"}.issubset(df.columns):
                ax.plot(df["Date"], df["prediction"], label=model, linestyle="-")
                ax.plot(
                    df["Date"],
                    df["actual"],
                    label=f"Actual ({model})",
                    linestyle="--",
                    alpha=0.5,
                )

    ax.set_title("Volatility Model Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


# --------------------------- Sidebar UI --------------------------- #
st.sidebar.header("Select Models to Compare")
selected_models = st.sidebar.multiselect(
    "Available Models:",
    options=list(model_files.keys()),
    default=["GARCH", "LSTM", "EWMA"],
)

if selected_models:
    plot_models(selected_models)
else:
    st.info("Please select at least one model to view the plot.")
