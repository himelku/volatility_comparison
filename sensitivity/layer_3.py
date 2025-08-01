import pandas as pd
import numpy as np
import os
import sys
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Add project root to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from garch.garch_functions import train_garch_model, garch_data_prep
from sensitivity.sensitivity_model_function import create_model_sensitivity
from lstm.lstm_functions import create_dataset, should_retrain

# ------------------ Load the Datasets ------------------ #
data_dir = os.path.join("data")
os.makedirs(data_dir, exist_ok=True)

# SPY LSTM data
sp_lstm_path = os.path.join(data_dir, "SPY_15min_lstm.csv")
sp_lstm = pd.read_csv(sp_lstm_path)
sp_lstm["Date"] = pd.to_datetime(sp_lstm["Date"])

# SPY raw data for GARCH
sp_path = os.path.join(data_dir, "SPY_15min_intraday.csv")
sp = pd.read_csv(sp_path)
sp["Date"] = pd.to_datetime(sp["timestamp"])

# VIX data
vix_path = os.path.join(data_dir, "vix_15min.csv")
vix = pd.read_csv(vix_path)
vix["Date"] = pd.to_datetime(vix["timestamp"])

# ------------------ VIX ------------------ #
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

# ------------------ GARCH ------------------ #
sp_garch = garch_data_prep(sp)
start_date = sp_garch["Date"].min().strftime("%Y-%m-%d %H:%M:%S")
garch_results = train_garch_model(sp_garch, start_date)

# Merge with LSTM and VIX datasets
sp_garch = pd.merge(sp_lstm, garch_results, on=["Date"], how="left")
sp_garch = sp_garch.rename(columns={"prediction": "predicted_volatility_garch"})
sp_garch_vix = pd.merge_asof(
    sp_garch.sort_values("Date").reset_index(drop=True),
    vix.sort_values("Date").reset_index(drop=True),
    on="Date",
    direction="backward",
)

# df = sp_garch_vix
df = sp_garch_vix.dropna(subset=['volatility', 'Close_vix']).copy()

# ------------------ LSTM-GARCH-VIX with 3 layers ------------------ #
# Setup features and target column
feature_columns = [
    col for col in df.columns if col not in ["volatility", "Date", "timestamp"]
]
target_column = "volatility"

# ------------------ Define Parameters ------------------ #
time_steps = 22  # ~5.5 hours of 15-min bars
steps_per_day = 26
initial_train_size = 21 * steps_per_day  # ~1 month
validation_size = 7 * steps_per_day  # ~1 week

X, y = create_dataset(
    df[feature_columns], df[target_column].values.reshape(-1, 1), time_steps
)

input_shape = (time_steps, X.shape[2])
model_save_path = os.path.join(data_dir, "lstm_garch_vix_layer_3_intraday.weights.h5")

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

results = []
counter = 0

# ------------------ Walk-Forward Prediction ------------------ #
for i in range(len(df) - initial_train_size - validation_size - 1):
    if i + initial_train_size + validation_size + time_steps > len(X):
        print(
            "Not enough data to form a complete sequence for testing. Ending predictions."
        )
        break

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit scaler on current training data
    scaler_X.fit(X[i : i + initial_train_size].reshape(-1, X.shape[2]))
    scaler_y.fit(y[i : i + initial_train_size].reshape(-1, 1))

    # Transform train, validation, and test data
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

    # Train or update the model
    if should_retrain(counter) or not os.path.exists(model_save_path):
        model = create_model_sensitivity(input_shape, lstm_layers=3)
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
        model = create_model_sensitivity(input_shape, lstm_layers=3)
        model.load_weights(model_save_path)
        model.fit(
            train_X[-1].reshape(1, *train_X[-1].shape),
            train_y[-1].reshape(1, 1),
            epochs=1,
            verbose=0,
        )

    predicted = model.predict(test_X)
    # Inverse transform
    predicted = scaler_y.inverse_transform(predicted.reshape(-1, 1))
    actual = scaler_y.inverse_transform(test_y.reshape(-1, 1))

    # Defensive check
    if np.isnan(actual).any() or np.isnan(predicted).any():
        print("⚠️ Warning: NaNs detected in actual or predicted values!")
        print(f"Actual: {actual.flatten()}")
        print(f"Predicted: {predicted.flatten()}")
        continue  # Skip this test point

    # Compute MAE
    mae = mean_absolute_error(actual, predicted)

    # Print current results
    current_result = {
        "train_start": df["Date"].iloc[i + time_steps],
        "train_end": df["Date"].iloc[i + initial_train_size + time_steps - 1],
        "validation_start": df["Date"].iloc[i + initial_train_size + time_steps],
        "validation_end": df["Date"].iloc[
            i + initial_train_size + validation_size + time_steps - 1
        ],
        "test_date": df["Date"].iloc[
            i + initial_train_size + validation_size + time_steps
        ],
        "prediction": predicted.flatten()[0],
        "actual": actual.flatten()[0],
        "mae": mae,
    }
    print(current_result)

    results.append(current_result)
    counter += 1

# Save results
results_path = os.path.join(data_dir, "results_lstm_garch_vix_layer_3_intraday.csv")
pd.DataFrame(results).to_csv(results_path, index=False)
print(f"Results saved to {results_path}")
