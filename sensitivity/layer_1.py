import pandas as pd
import numpy as np
import os
import sys
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import custom functions
from garch.garch_functions import train_garch_model, garch_data_prep
from sensitivity.sensitivity_model_function import create_model_sensitivity
from lstm.lstm_functions import create_dataset, should_retrain

# ------------------ Load Datasets ------------------ #
# Load SPY LSTM data
spy_lstm_path = os.path.join("data", "SPY_15min_lstm.csv")
spy_lstm = pd.read_csv(spy_lstm_path)
spy_lstm["Date"] = pd.to_datetime(spy_lstm["Date"])

# Load SPY raw data for GARCH
spy_raw_path = os.path.join("data", "SPY_15min_intraday.csv")
spy_raw = pd.read_csv(spy_raw_path)
spy_raw["Date"] = pd.to_datetime(spy_raw["timestamp"])

# Load VIX data
vix_path = os.path.join("data", "vix_15min.csv")
vix = pd.read_csv(vix_path)
vix["Date"] = pd.to_datetime(vix["timestamp"])

# ------------------ Preprocess VIX ------------------ #
if "Close" in vix.columns:
    vix = vix.rename(columns={"Close": "Close_vix"})
elif "close" in vix.columns:
    vix = vix.rename(columns={"close": "Close_vix"})
else:
    raise KeyError("VIX file is missing 'Close' or 'close' column needed for merging.")
vix.drop(
    columns=["timestamp", "Open", "High", "Low", "Adj Close", "Volume"],
    errors="ignore",
    inplace=True,
)

# ------------------ GARCH Modeling ------------------ #
spy_garch = garch_data_prep(spy_raw)
start_date = spy_garch["Date"].min().strftime("%Y-%m-%d %H:%M:%S")
garch_results = train_garch_model(spy_garch, start_date)

# Merge with LSTM and VIX dataset
merged = pd.merge(spy_lstm, garch_results, on="Date", how="left")
merged = merged.rename(columns={"prediction": "predicted_volatility_garch"})
merged = pd.merge_asof(
    merged.sort_values("Date").reset_index(drop=True),
    vix.sort_values("Date").reset_index(drop=True),
    on="Date",
    direction="backward",
)

# Drop rows with missing essential columns
essential_cols = ["volatility", "predicted_volatility_garch", "Close_vix"]
merged.dropna(subset=essential_cols, inplace=True)
print(f"Merged dataset shape: {merged.shape}")

# ------------------ Feature Engineering ------------------ #
feature_columns = [
    col
    for col in merged.columns
    if col not in ["volatility", "Date", "timestamp"]
    and pd.api.types.is_numeric_dtype(merged[col])
]
target_column = "volatility"

# ------------------ Dataset Setup ------------------ #
time_steps = 22  # ~5.5 hours of 15-min bars
steps_per_day = 26  # approx. 6.5 hours/day
initial_train_size = 21 * steps_per_day  # ~1 month
validation_size = 7 * steps_per_day  # ~1 week

X, y = create_dataset(
    merged[feature_columns], merged[target_column].values.reshape(-1, 1), time_steps
)

input_shape = (time_steps, X.shape[2])
model_save_path = "lstm_garch_vix_layer_1_intraday.weights.h5"
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# ------------------ Walk-Forward Validation ------------------ #
results = []
counter = 0

for i in range(len(merged) - initial_train_size - validation_size - 1):
    if i + initial_train_size + validation_size + time_steps > len(X):
        print("Not enough data to form a complete test sequence. Ending.")
        break

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit scalers on training data
    scaler_X.fit(X[i : i + initial_train_size].reshape(-1, X.shape[2]))
    scaler_y.fit(y[i : i + initial_train_size].reshape(-1, 1))

    # Transform train, validation, and test sets
    train_X = scaler_X.transform(
        X[i : i + initial_train_size].reshape(-1, X.shape[2])
    ).reshape(-1, time_steps, X.shape[2])
    train_y = scaler_y.transform(y[i : i + initial_train_size].reshape(-1, 1)).reshape(
        -1, 1
    )

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

    # Train or update model
    if should_retrain(counter) or not os.path.exists(model_save_path):
        model = create_model_sensitivity(input_shape, lstm_layers=1)
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
        model = create_model_sensitivity(input_shape, lstm_layers=1)
        model.load_weights(model_save_path)
        model.fit(
            train_X[-1].reshape(1, *train_X[-1].shape),
            train_y[-1].reshape(1, 1),
            epochs=1,
            verbose=0,
        )

    # Predict and inverse transform
    predicted = model.predict(test_X)
    predicted = scaler_y.inverse_transform(predicted.reshape(-1, 1))
    actual = scaler_y.inverse_transform(test_y.reshape(-1, 1))
    mae = mean_absolute_error(actual, predicted)

    current_result = {
        "train_start": merged["Date"].iloc[i + time_steps],
        "train_end": merged["Date"].iloc[i + initial_train_size + time_steps - 1],
        "validation_start": merged["Date"].iloc[i + initial_train_size + time_steps],
        "validation_end": merged["Date"].iloc[
            i + initial_train_size + validation_size + time_steps - 1
        ],
        "test_date": merged["Date"].iloc[
            i + initial_train_size + validation_size + time_steps
        ],
        "prediction": float(predicted.flatten()[0]),
        "actual": float(actual.flatten()[0]),
        "mae": mae,
    }
    print(current_result)
    results.append(current_result)
    counter += 1

# Save results
results_dir = os.path.join("data")
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, "results_lstm_garch_vix_layer_1_intraday.csv")
pd.DataFrame(results).to_csv(results_path, index=False)
print(f"Results saved to {results_path}")
