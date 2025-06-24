import pandas as pd
import numpy as np
import os
import sys
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from garch.garch_functions import train_garch_model, garch_data_prep
from lstm.lstm_functions import create_dataset, should_retrain
from ewma_functions import compute_ewma_volatility

# ------------------ Load Datasets ------------------ #
data_dir = os.path.join("data")
spy_lstm = pd.read_csv(os.path.join(data_dir, "SPY_15min_lstm.csv"))
spy_lstm["Date"] = pd.to_datetime(spy_lstm["Date"])

spy_raw = pd.read_csv(os.path.join(data_dir, "SPY_15min_intraday.csv"))
spy_raw["Date"] = pd.to_datetime(spy_raw["timestamp"])
spy_raw["log_returns"] = np.log(spy_raw["close"] / spy_raw["close"].shift(1))
spy_raw["log_returns_scaled"] = 1000 * spy_raw["log_returns"]
spy_raw["ewma_volatility"] = compute_ewma_volatility(spy_raw["log_returns"])

# ------------------ GARCH Modeling ------------------ #
spy_garch = garch_data_prep(spy_raw)
spy_garch["log_returns"] = spy_raw["log_returns_scaled"]
start_date = spy_garch["Date"].min().strftime("%Y-%m-%d %H:%M:%S")
garch_results = train_garch_model(spy_garch, start_date)
print("Columns in GARCH results:", garch_results.columns)

# Rename column if needed
if "prediction" in garch_results.columns:
    garch_results.rename(
        columns={"prediction": "predicted_volatility_garch"}, inplace=True
    )

# ------------------ Load VIX ------------------ #
vix = pd.read_csv(os.path.join(data_dir, "vix_15min.csv"))
vix["Date"] = pd.to_datetime(vix["timestamp"])
vix = vix.rename(
    columns={"Close": "Close_vix"} if "Close" in vix.columns else {"close": "Close_vix"}
)
vix.drop(
    columns=["timestamp", "Open", "High", "Low", "Adj Close", "Volume"],
    errors="ignore",
    inplace=True,
)

# ------------------ Merge ------------------ #
merged = pd.merge(
    spy_lstm,
    garch_results[["Date", "predicted_volatility_garch"]],
    on="Date",
    how="left",
)
merged = pd.merge(merged, spy_raw[["Date", "ewma_volatility"]], on="Date", how="left")
merged = pd.merge_asof(
    merged.sort_values("Date"), vix.sort_values("Date"), on="Date", direction="backward"
)

# Clean and prepare
essential_cols = [
    "volatility",
    "predicted_volatility_garch",
    "Close_vix",
    "ewma_volatility",
]
merged.dropna(subset=essential_cols, inplace=True)

# ------------------ Features & Target ------------------ #
feature_columns = [
    col for col in merged.columns if col not in ["volatility", "Date", "timestamp"]
]
target_column = "volatility"

# ------------------ LSTM Settings ------------------ #
time_steps = 22
steps_per_day = 26
initial_train_size = 21 * steps_per_day
validation_size = 7 * steps_per_day

X, y = create_dataset(
    merged[feature_columns], merged[target_column].values.reshape(-1, 1), time_steps
)
input_shape = (time_steps, X.shape[2])

model_save_path = os.path.join("data", "lstm_garch_ewma_vix.weights.h5")
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

results = []
counter = 0

# ------------------ Walk-Forward Validation ------------------ #
for i in range(len(merged) - initial_train_size - validation_size - 1):
    if i + initial_train_size + validation_size + time_steps > len(X):
        break

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_X = scaler_X.fit_transform(
        X[i : i + initial_train_size].reshape(-1, X.shape[2])
    ).reshape(-1, time_steps, X.shape[2])
    train_y = scaler_y.fit_transform(
        y[i : i + initial_train_size].reshape(-1, 1)
    ).reshape(-1, 1)
    val_X = scaler_X.transform(
        X[i + initial_train_size : i + initial_train_size + validation_size].reshape(
            -1, X.shape[2]
        )
    ).reshape(-1, time_steps, X.shape[2])
    val_y = scaler_y.transform(
        y[i + initial_train_size : i + initial_train_size + validation_size].reshape(
            -1, 1
        )
    ).reshape(-1, 1)
    test_X = scaler_X.transform(
        X[
            i
            + initial_train_size
            + validation_size : i
            + initial_train_size
            + validation_size
            + 1
        ].reshape(-1, X.shape[2])
    ).reshape(-1, time_steps, X.shape[2])
    test_y = scaler_y.transform(
        y[
            i
            + initial_train_size
            + validation_size : i
            + initial_train_size
            + validation_size
            + 1
        ].reshape(-1, 1)
    ).reshape(-1, 1)

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")

    if should_retrain(counter) or not os.path.exists(model_save_path):
        model.fit(
            train_X,
            train_y,
            epochs=100,
            batch_size=64,
            validation_data=(val_X, val_y),
            verbose=0,
            callbacks=[early_stopping],
        )
        model.save_weights(model_save_path)
    else:
        model.load_weights(model_save_path)
        model.fit(
            train_X[-1].reshape(1, *train_X[-1].shape),
            train_y[-1].reshape(1, 1),
            epochs=1,
            verbose=0,
        )

    predicted = model.predict(test_X)
    predicted = scaler_y.inverse_transform(predicted.reshape(-1, 1))
    actual = scaler_y.inverse_transform(test_y.reshape(-1, 1))
    mae = mean_absolute_error(actual, predicted)

    results.append(
        {
            "train_start": merged["Date"].iloc[i + time_steps],
            "train_end": merged["Date"].iloc[i + initial_train_size + time_steps - 1],
            "validation_start": merged["Date"].iloc[
                i + initial_train_size + time_steps
            ],
            "validation_end": merged["Date"].iloc[
                i + initial_train_size + validation_size + time_steps - 1
            ],
            "test_date": merged["Date"].iloc[
                i + initial_train_size + validation_size + time_steps
            ],
            "prediction": predicted.flatten()[0],
            "actual": actual.flatten()[0],
            "mae": mae,
        }
    )

    counter += 1

# Save results
results_df = pd.DataFrame(results)
results_path = os.path.join(data_dir, "results_lstm_garch_ewma_vix_intraday.csv")
results_df.to_csv(results_path, index=False)
print(f"✅ Results saved to {results_path}")
