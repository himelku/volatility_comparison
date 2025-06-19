import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# ------------------ Helper Functions ------------------ #
def load_data(filepath):
    df = pd.read_csv(filepath)
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

# ------------------ Sidebar ------------------ #
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

view_mode = st.sidebar.radio(
    "Choose view mode:", ["Scroll View", "Compare on One Chart"]
)

metric_choice = st.sidebar.radio("Metric for Best Model:", ["MAE", "RMSE"])

# ------------------ Data Preparation ------------------ #
model_data = {}
metrics = {}

for model_name in selected_models:
    file_path = os.path.join(data_dir, model_files[model_name])
    if not os.path.exists(file_path):
        st.sidebar.warning(f"File not found: {model_name}")
        continue

    df = load_data(file_path)

    # Special handling
    if (
        model_name == "EWMA"
        and "ewma_volatility" in df.columns
        and "log_returns" in df.columns
    ):
        df["rolling_std"] = df["log_returns"].rolling(26).std()
        df.dropna(inplace=True)
        actual = df["rolling_std"]
        predicted = df["ewma_volatility"]
        df.rename(
            columns={"ewma_volatility": "prediction", "rolling_std": "actual"},
            inplace=True,
        )
    elif (
        model_name == "LSTM-GARCH-EWMA-VIX"
        and "predicted_volatility_lstm_garch_ewma_vix" in df.columns
    ):
        df.rename(
            columns={"predicted_volatility_lstm_garch_ewma_vix": "prediction"},
            inplace=True,
        )
    elif "prediction" not in df.columns or "actual" not in df.columns:
        continue

    df = df.dropna(subset=["prediction", "actual"])
    mae, rmse = compute_metrics(df["actual"], df["prediction"])

    model_data[model_name] = df
    metrics[model_name] = {"MAE": mae, "RMSE": rmse}

# ------------------ Best Model Identification ------------------ #
best_model = None
if metrics:
    best_model = min(metrics.items(), key=lambda x: x[1][metric_choice])[0]

# ------------------ Scroll View ------------------ #
if view_mode == "Scroll View":
    for model_name in selected_models:
        if model_name not in model_data:
            continue
        df = model_data[model_name]
        mae = metrics[model_name]["MAE"]
        rmse = metrics[model_name]["RMSE"]

        st.markdown(f"### {model_name}")
        if model_name == best_model:
            st.success(f"🏆 Best Model Based on {metric_choice}: {model_name}")

        st.write(f"**MAE**: {mae:.6f} | **RMSE**: {rmse:.6f}")
        fig = plot_interactive(
            [df[["Date", "prediction"]], df[["Date", "actual"]]],
            [f"Predicted ({model_name})", "Actual"],
            f"{model_name} vs. Actual Volatility",
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------ One Chart View ------------------ #
else:
    st.subheader("📊 Combined Comparison Chart")
    df_list = []
    labels = []

    for model_name in selected_models:
        if model_name in model_data:
            df = model_data[model_name][["Date", "prediction"]].copy()
            df.rename(columns={"prediction": model_name}, inplace=True)
            df_list.append(df)
            labels.append(model_name)

    if df_list:
        df_merged = df_list[0]
        for df in df_list[1:]:
            df_merged = pd.merge(df_merged, df, on="Date", how="outer")
        df_merged = df_merged.sort_values("Date").dropna()

        combined_df_list = [df_merged[["Date", model]] for model in labels]
        fig = plot_interactive(
            combined_df_list, labels, "Volatility Comparison Across Models"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Show metrics table
    st.markdown("### 📌 Model Metrics")
    # Collect results in a list of dicts
    metrics_table = []
    for model_name in selected_models:
        if model_name in metrics:
            mae = metrics[model_name]["MAE"]
            rmse = metrics[model_name]["RMSE"]
            is_best = "🏆" if model_name == best_model else ""
            metrics_table.append(
                {
                    "Model": model_name,
                    "MAE": round(mae, 6),
                    "RMSE": round(rmse, 6),
                    "Best": is_best,
                }
            )

    # Convert to DataFrame and display as a table
    metrics_df = pd.DataFrame(metrics_table)
    st.table(
        metrics_df
    )  # or use st.dataframe(metrics_df) for scrollable/sortable version
