import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# ------------------ Helper Functions ------------------ #
def load_data(filepath):
    df = pd.read_csv(filepath)
    # Assign standard datetime column
    for col in ["Date", "timestamp", "test_date"]:
        if col in df.columns:
            df["Date"] = pd.to_datetime(df[col])
            break
    return df


def compute_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse


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


# ------------------ Streamlit Config ------------------ #
st.set_page_config(layout="wide", page_title="Volatility Forecasting Dashboard")
st.title("📈 Volatility Forecasting Dashboard")

# ------------------ Sidebar Model Selector ------------------ #
data_dir = "data"
model_files = {
    "GARCH": "results_garch_intraday.csv",
    "LSTM": "results_lstm_intraday.csv",
    "LSTM-GARCH": "results_lstm_garch_intraday.csv",
    "LSTM-GARCH-VIX": "results_lstm_garch_vix_intraday.csv",
    "LSTM-GARCH-EWMA-VIX": "results_lstm_garch_ewma_vix_intraday.csv",
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

# ------------------ Main Area: Model Results ------------------ #
st.subheader("📊 Model Output Comparison")

for model_name in selected_models:
    file_path = os.path.join(data_dir, model_files[model_name])

    if not os.path.exists(file_path):
        st.warning(f"File not found for {model_name}: {file_path}")
        continue

    df = load_data(file_path)

    # ------------------ EWMA Case ------------------ #
    if model_name == "EWMA":
        if "ewma_volatility" in df.columns and "log_returns" in df.columns:
            df["rolling_std"] = df["log_returns"].rolling(26).std()
            df.dropna(inplace=True)
            mae, rmse = compute_metrics(df["rolling_std"], df["ewma_volatility"])

            st.markdown(f"### {model_name}")
            st.write(f"MAE: `{mae:.6f}` | RMSE: `{rmse:.6f}`")

            fig = plot_interactive(
                [df[["Date", "ewma_volatility"]], df[["Date", "rolling_std"]]],
                ["EWMA Volatility", "Rolling Std Dev (26)"],
                "EWMA vs Rolling Std Dev",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Required columns missing for {model_name}")
        continue

    # ------------------ LSTM-GARCH-EWMA-VIX Case ------------------ #
    elif model_name == "LSTM-GARCH-EWMA-VIX":
        if (
            "predicted_volatility_lstm_garch_ewma_vix" in df.columns
            and "actual" in df.columns
        ):
            df = df.dropna(
                subset=["predicted_volatility_lstm_garch_ewma_vix", "actual"]
            )
            df.rename(
                columns={"predicted_volatility_lstm_garch_ewma_vix": "prediction"},
                inplace=True,
            )
        else:
            st.warning(f"Required columns not found in {model_name}")
            continue

    # ------------------ Default Prediction-Based Models ------------------ #
    elif "prediction" in df.columns and "actual" in df.columns:
        df = df.dropna(subset=["prediction", "actual"])
    else:
        st.warning(f"Required columns not found in {model_name}")
        continue

    # ------------------ Metric & Plot Display ------------------ #
    mae, rmse = compute_metrics(df["actual"], df["prediction"])
    st.markdown(f"### {model_name}")
    st.write(f"MAE: `{mae:.6f}` | RMSE: `{rmse:.6f}`")

    fig = plot_interactive(
        [df[["Date", "prediction"]], df[["Date", "actual"]]],
        [f"Predicted ({model_name})", "Actual"],
        f"{model_name} vs. Actual Volatility",
    )
    st.plotly_chart(fig, use_container_width=True)
