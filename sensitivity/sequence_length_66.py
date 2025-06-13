import pandas as pd
import numpy as np
import os
import sys
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras import Input
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from garch.garch_functions import train_garch_model, garch_data_prep
from sensitivity.sensitivity_model_function import create_model_sensitivity  # Optional override
from lstm.lstm_functions import create_dataset, should_retrain

# ------------------ Define LSTM Model ------------------ #
def create_model_sensitivity(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model

# ------------------ Load Datasets ------------------ #
data_dir = os.path.join("data")
os.makedirs(data_dir, exist_ok=True)

sp_lstm_path = os.path.join(data_dir, "SPY_15min_lstm.csv")
sp_lstm = pd.read_csv(sp_lstm_path)
sp_lstm["Date"] = pd.to_datetime(sp_lstm["Date"])

sp_path = os.path.join(data_dir, "SPY_15min_intraday.csv")
sp = pd.read_csv(sp_path)
sp["Date"] = pd.to_datetime(sp["timestamp"])

# Compute log returns if not already present
if "log_returns" not in sp.columns:
    if "close" not in sp.columns and "Close" in sp.columns:
        sp["close"] = sp["Close"]
    sp = sp.sort_values("Date").reset_index(drop=True)
    sp["log_returns"] = np.log(sp["close"] / sp["close"].shift(1))
sp.dropna(subset=["log_returns"], inplace=True)
sp["log_returns"] *= 1000  # Rescale for GARCH stability

# Load VIX data
vix_path = os.path.join(data_dir, "vix_15min.csv")
vix = pd.read_csv(vix_path)
vix["Date"] = pd.to_datetime(vix["timestamp"])
if "Close" in vix.columns:
    vix = vix.rename(columns={"Close": "Close_vix"})
elif "close" in vix.columns:
    vix = vix.rename(columns={"close": "Close_vix"})
else:
    raise KeyError("VIX file must contain 'Close' or 'close' column.")
vix.drop(columns=["timestamp", "open", "high", "low", "volume"], errors="ignore", inplace=True)

# ------------------ GARCH Modeling ------------------ #
sp_garch = garch_data_prep(sp)
start_date = sp_garch["Date"].min().strftime("%Y-%m-%d %H:%M:%S")
garch_results = train_garch_model(sp_garch, start_date)

# Merge dataframes
sp_garch = pd.merge(sp_lstm, garch_results, on="Date", how="left")
sp_garch = sp_garch.rename(columns={"prediction": "predicted_volatility_garch"})
sp_garch_vix = pd.merge_asof(
    sp_garch.sort_values("Date").reset_index(drop=True),
    vix.sort_values("Date").reset_index(drop=True),
    on="Date",
    direction="backward"
)

df = sp_garch_vix

# ------------------ Drop rows with NaNs in important columns ------------------ #
df = df.dropna(subset=["volatility", "predicted_volatility_garch", "log_returns", "Close_vix"]).reset_index(drop=True)

# ------------------ Setup Features and Parameters ------------------ #
feature_columns = [col for col in df.columns if col not in ["volatility", "Date", "timestamp"]]
target_column = "volatility"

# Clip outliers in features
df[feature_columns] = df[feature_columns].clip(lower=-1e5, upper=1e5)

time_steps = 66
steps_per_day = 26
initial_train_size = 21 * steps_per_day
validation_size = 7 * steps_per_day

X, y = create_dataset(df[feature_columns], df[target_column].values.reshape(-1, 1), time_steps)
input_shape = (time_steps, X.shape[2])

model_save_path = os.path.join(data_dir, "lstm_garch_vix_seq_length_66.weights.h5")
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

results = []
counter = 0

# ------------------ Walk-Forward Training ------------------ #
for i in range(len(df) - initial_train_size - validation_size - 1):
    if i + initial_train_size + validation_size + time_steps > len(X):
        print("Not enough data to form a complete sequence for testing. Ending predictions.")
        break

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    try:
        train_X = scaler_X.fit_transform(X[i:i + initial_train_size].reshape(-1, X.shape[2])).reshape(-1, time_steps, X.shape[2])
        train_y = scaler_y.fit_transform(y[i:i + initial_train_size].reshape(-1, 1)).reshape(-1, 1)
        val_X = scaler_X.transform(X[i + initial_train_size:i + initial_train_size + validation_size].reshape(-1, X.shape[2])).reshape(-1, time_steps, X.shape[2])
        val_y = scaler_y.transform(y[i + initial_train_size:i + initial_train_size + validation_size].reshape(-1, 1)).reshape(-1, 1)
        test_X = scaler_X.transform(X[i + initial_train_size + validation_size:i + initial_train_size + validation_size + 1].reshape(-1, X.shape[2])).reshape(-1, time_steps, X.shape[2])
        test_y = scaler_y.transform(y[i + initial_train_size + validation_size:i + initial_train_size + validation_size + 1].reshape(-1, 1)).reshape(-1, 1)
    except ValueError as e:
        print(f"Scaler error at iteration {i}: {e}")
        continue

    if np.isnan(train_X).any() or np.isnan(train_y).any():
        print(f"NaNs in training data at iteration {i}, skipping.")
        continue
    if np.std(train_y) < 1e-8:
        print(f"No variance in train_y at iteration {i}, skipping.")
        continue

    if should_retrain(counter) or not os.path.exists(model_save_path):
        model = create_model_sensitivity(input_shape)
        model.fit(train_X, train_y, epochs=100, batch_size=64, validation_data=(val_X, val_y), verbose=0, callbacks=[early_stopping])
        model.save_weights(model_save_path)
    else:
        model = create_model_sensitivity(input_shape)
        model.load_weights(model_save_path)
        model.fit(train_X[-1].reshape(1, *train_X[-1].shape), train_y[-1].reshape(1, 1), epochs=1, verbose=0)

    if np.isnan(test_X).any():
        print(f"NaN in test_X at iteration {i}, skipping.")
        continue

    predicted = model.predict(test_X)
    if np.isnan(predicted).any():
        print(f"NaN in prediction at iteration {i}, skipping.")
        continue

    predicted = scaler_y.inverse_transform(predicted.reshape(-1, 1))
    actual = scaler_y.inverse_transform(test_y.reshape(-1, 1))

    if np.isnan(actual).any() or np.isnan(predicted).any():
        print(f"NaN after inverse scaling at iteration {i}, skipping.")
        continue

    mae = mean_absolute_error(actual, predicted)

    result = {
        "train_start": df["Date"].iloc[i + time_steps],
        "train_end": df["Date"].iloc[i + initial_train_size + time_steps - 1],
        "validation_start": df["Date"].iloc[i + initial_train_size + time_steps],
        "validation_end": df["Date"].iloc[i + initial_train_size + validation_size + time_steps - 1],
        "test_date": df["Date"].iloc[i + initial_train_size + validation_size + time_steps],
        "prediction": predicted.flatten()[0],
        "actual": actual.flatten()[0],
        "mae": mae,
    }
    print(result)
    results.append(result)
    counter += 1

# ------------------ Save Results ------------------ #
results_path = os.path.join(data_dir, "results_lstm_garch_vix_lookback_66.csv")
pd.DataFrame(results).to_csv(results_path, index=False)
print(f"\n✅ Results saved to: {results_path}")
