import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------ Configuration ------------------ #
data_dir = os.path.join("data")
st.set_page_config(page_title="Volatility Model Dashboard", layout="wide")

# ------------------ Utility Functions ------------------ #
def load_csv(filename):
    path = os.path.join(data_dir, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        for col in ["Date", "timestamp", "test_date"]:
            if col in df.columns:
                df["Date"] = pd.to_datetime(df[col])
                break
        return df.dropna()
    return None

def plot_predictions(df, title, pred_col="prediction", actual_col="actual"):
    if df is None or pred_col not in df.columns or actual_col not in df.columns:
        st.warning(f"Missing columns for: {title}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df[pred_col], label="Predicted", color="royalblue")
    ax.plot(df["Date"], df[actual_col], label="Actual", linestyle="--", color="red")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    mae = mean_absolute_error(df[actual_col], df[pred_col])
    rmse = np.sqrt(mean_squared_error(df[actual_col], df[pred_col]))
    st.markdown(f"**MAE:** {mae:.6f} | **RMSE:** {rmse:.6f}")


def plot_ewma_vs_std(ewma_df):
    ewma_df["rolling_std"] = ewma_df["log_returns"].rolling(26).std()
    ewma_df = ewma_df.dropna()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ewma_df["Date"], ewma_df["ewma_volatility"], label="EWMA Volatility", color="blue")
    ax.plot(ewma_df["Date"], ewma_df["rolling_std"], label="Rolling Std Dev (26)", linestyle="--", color="gray")
    ax.set_title("EWMA vs Rolling Std Dev")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    mae = mean_absolute_error(ewma_df["rolling_std"], ewma_df["ewma_volatility"])
    rmse = np.sqrt(mean_squared_error(ewma_df["rolling_std"], ewma_df["ewma_volatility"]))
    st.markdown(f"**MAE:** {mae:.6f} | **RMSE:** {rmse:.6f}")

def plot_ewma_vs_vix():
    vix = load_csv("vix_15min.csv")
    ewma = load_csv("results_ewma.csv")
    if vix is not None and ewma is not None:
        merged = pd.merge(ewma, vix, on="Date")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(merged["Date"], merged["ewma_volatility"], label="EWMA Volatility", color="blue")
        ax.plot(merged["Date"], merged["close"], label="VIX Close", linestyle="--", color="orange")
        ax.set_title("EWMA Volatility vs VIX")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# ------------------ Streamlit Tabs ------------------ #
st.title("📈 Volatility Forecasting Models Dashboard")
tabs = st.tabs(["GARCH", "LSTM", "LSTM-GARCH", "LSTM-GARCH-VIX", "EWMA", "EWMA vs VIX"])

with tabs[0]:
    st.header("GARCH Predictions")
    df = load_csv("results_garch_intraday.csv")
    plot_predictions(df, "GARCH Forecast vs Actual")

with tabs[1]:
    st.header("LSTM Predictions")
    df = load_csv("results_lstm_intraday.csv")
    plot_predictions(df, "LSTM Forecast vs Actual")

with tabs[2]:
    st.header("LSTM-GARCH Predictions")
    df = load_csv("results_lstm_garch_intraday.csv")
    plot_predictions(df, "LSTM-GARCH Forecast vs Actual")

with tabs[3]:
    st.header("LSTM-GARCH-VIX Predictions")
    df = load_csv("results_lstm_garch_vix_intraday.csv")
    plot_predictions(df, "LSTM-GARCH-VIX Forecast vs Actual")

with tabs[4]:
    st.header("EWMA Volatility")
    df = load_csv("results_ewma.csv")
    if df is not None and {"log_returns", "ewma_volatility"}.issubset(df.columns):
        plot_ewma_vs_std(df)
    else:
        st.warning("Missing EWMA columns.")

with tabs[5]:
    st.header("EWMA vs VIX")
    plot_ewma_vs_vix()
