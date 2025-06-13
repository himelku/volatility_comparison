import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# ------------------ Helper Functions ------------------ #
def load_data(file):
    df = pd.read_csv(file)
    if "Date" not in df.columns:
        for col in ["timestamp", "test_date"]:
            if col in df.columns:
                df["Date"] = pd.to_datetime(df[col])
                break
    else:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def plot_interactive(df_list, labels, title):
    fig = go.Figure()
    for df, label in zip(df_list, labels):
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df[df.columns[1]],
                mode="lines",
                name=label,
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.6f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Volatility",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def compute_metrics(df):
    mae = mean_absolute_error(df["actual"], df["prediction"])
    rmse = np.sqrt(mean_squared_error(df["actual"], df["prediction"]))
    return mae, rmse


# ------------------ App Config ------------------ #
st.set_page_config(layout="wide", page_title="Volatility Model Dashboard")
st.title("📈 Volatility Forecasting Dashboard")

# ------------------ Sidebar ------------------ #
data_dir = "data"
model_files = {
    "GARCH": "results_garch_intraday.csv",
    "LSTM": "results_lstm_intraday.csv",
    "LSTM-GARCH": "results_lstm_garch_intraday.csv",
    "LSTM-GARCH-VIX": "results_lstm_garch_vix_intraday.csv",
    "LSTM-GARCH-VIX-ReLU": "results_lstm_garch_vix_relu.csv",
    "LSTM-GARCH-VIX-MAE Loss": "results_lstm_garch_vix_mae_loss_intraday.csv",
    "LSTM-GARCH-VIX-Pct Change": "results_lstm_garch_vix_pct_change.csv",
    "LSTM-GARCH-VIX-Lookback 5": "results_lstm_garch_vix_lookback_5.csv",
    "LSTM-GARCH-VIX-Lookback 66": "results_lstm_garch_vix_lookback_66.csv",
    "LSTM-GARCH-VIX-1 Layer": "results_lstm_garch_vix_layer_1_intraday.csv",
    "LSTM-GARCH-VIX-3 Layers": "results_lstm_garch_vix_layer_3_intraday.csv",
    "EWMA": "results_ewma.csv",
}

selected_models = st.sidebar.multiselect(
    "Select models to visualize:",
    options=list(model_files.keys()),
    default=["GARCH", "LSTM"],
)

# ------------------ Main Tab ------------------ #
st.subheader("📊 Model Output Comparison")
for model_name in selected_models:
    file_path = os.path.join(data_dir, model_files[model_name])
    if not os.path.exists(file_path):
        st.warning(f"File not found for {model_name}: {file_path}")
        continue

    df = load_data(file_path)
    if "actual" in df.columns and "prediction" in df.columns:
        df = df.dropna(subset=["actual", "prediction"])
        mae, rmse = compute_metrics(df)
        st.markdown(f"### {model_name}")
        st.write(f"MAE: `{mae:.6f}` | RMSE: `{rmse:.6f}`")

        chart = plot_interactive(
            [df[["Date", "prediction"]], df[["Date", "actual"]]],
            [f"Predicted ({model_name})", "Actual"],
            f"{model_name} vs. Actual Volatility",
        )
        st.plotly_chart(chart, use_container_width=True)

    elif (
        model_name == "EWMA"
        and "ewma_volatility" in df.columns
        and "log_returns" in df.columns
    ):
        df["rolling_std"] = df["log_returns"].rolling(26).std()
        df.dropna(inplace=True)
        mae = mean_absolute_error(df["rolling_std"], df["ewma_volatility"])
        rmse = np.sqrt(mean_squared_error(df["rolling_std"], df["ewma_volatility"]))
        st.markdown("### EWMA")
        st.write(f"MAE: `{mae:.6f}` | RMSE: `{rmse:.6f}`")

        chart = plot_interactive(
            [df[["Date", "ewma_volatility"]], df[["Date", "rolling_std"]]],
            ["EWMA Volatility", "Rolling Std Dev (26)"],
            "EWMA vs Rolling Std Dev",
        )
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.warning(f"Unsupported format in {model_name}")
