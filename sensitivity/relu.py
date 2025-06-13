import pandas as pd
import numpy as np
import os
import sys
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from garch.garch_functions import train_garch_model, garch_data_prep
from sensitivity.sensitivity_model_function import create_model_sensitivity
from lstm.lstm_functions import create_dataset, should_retrain

data_dir = os.path.join("data")
os.makedirs(data_dir, exist_ok=True)

# Load datasets
sp_lstm = pd.read_csv(os.path.join(data_dir, "SPY_15min_lstm.csv"))
sp_lstm["Date"] = pd.to_datetime(sp_lstm["Date"])

sp = pd.read_csv(os.path.join(data_dir, "SPY_15min_intraday.csv"))
sp["Date"] = pd.to_datetime(sp["timestamp"])

vix = pd.read_csv(os.path.join(data_dir, "vix_15min.csv"))
vix["Date"] = pd.to_datetime(vix["timestamp"])
vix = vix.rename(columns={"close": "Close_vix"})
vix.drop(columns=["open", "high", "low", "volume", "timestamp"], inplace=True)

sp_garch = garch_data_prep(sp)
start_date = sp_garch["Date"].min().strftime("%Y-%m-%d")
garch_results = train_garch_model(sp_garch, start_date)

sp_garch = pd.merge(sp_lstm, garch_results, on=["Date"], how="left")
sp_garch = sp_garch.rename(columns={"prediction": "predicted_volatility_garch"})
sp_garch_vix = pd.merge(sp_garch, vix, on=["Date"], how="left")

df = sp_garch_vix.dropna().reset_index(drop=True)

feature_columns = [col for col in df.columns if col not in ["volatility", "Date"]]
target_column = "volatility"

time_steps = 22
steps_per_day = 26
initial_train_size = 21 * steps_per_day
validation_size = 7 * steps_per_day

X, y = create_dataset(df[feature_columns], df[target_column].values.reshape(-1, 1), time_steps)
input_shape = (time_steps, X.shape[2])
model_save_path = os.path.join(data_dir, "lstm_garch_vix_relu.weights.h5")
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

results = []
counter = 0
skipped = 0

for i in range(len(df) - initial_train_size - validation_size - 1):
    if i + initial_train_size + validation_size + time_steps > len(X):
        break

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_reshaped = X[i:i+initial_train_size].reshape(-1, X.shape[2])
    y_train_reshaped = y[i:i+initial_train_size].reshape(-1, 1)

    if np.all(y_train_reshaped == y_train_reshaped[0]):
        print(f"Constant y detected at iteration {i}. Skipping...")
        skipped += 1
        continue

    if np.any(np.std(X_train_reshaped, axis=0) == 0):
        print(f"Constant X feature detected at iteration {i}. Skipping...")
        skipped += 1
        continue

    scaler_X.fit(X_train_reshaped)
    scaler_y.fit(y_train_reshaped)

    train_X = scaler_X.transform(X_train_reshaped).reshape(-1, time_steps, X.shape[2])
    train_y = scaler_y.transform(y_train_reshaped).reshape(-1, 1)

    val_X = scaler_X.transform(X[i+initial_train_size:i+initial_train_size+validation_size].reshape(-1, X.shape[2])).reshape(-1, time_steps, X.shape[2])
    val_y = scaler_y.transform(y[i+initial_train_size:i+initial_train_size+validation_size].reshape(-1, 1)).reshape(-1, 1)

    test_X = scaler_X.transform(X[i+initial_train_size+validation_size:i+initial_train_size+validation_size+1].reshape(-1, X.shape[2])).reshape(-1, time_steps, X.shape[2])
    test_y = scaler_y.transform(y[i+initial_train_size+validation_size:i+initial_train_size+validation_size+1].reshape(-1, 1)).reshape(-1, 1)

    model = create_model_sensitivity(input_shape, activation_function='relu')

    if should_retrain(counter) or not os.path.exists(model_save_path):
        model.fit(train_X, train_y, epochs=100, batch_size=64, validation_data=(val_X, val_y), verbose=0, callbacks=[early_stopping])
        model.save_weights(model_save_path)
    else:
        model.load_weights(model_save_path)
        model.fit(train_X[-1].reshape(1, *train_X[-1].shape), train_y[-1].reshape(1, 1), epochs=1, verbose=0)

    predicted = model.predict(test_X)
    predicted = scaler_y.inverse_transform(predicted.reshape(-1, 1))
    actual = scaler_y.inverse_transform(test_y.reshape(-1, 1))

    if np.isnan(actual).any() or np.isnan(predicted).any():
        print(f"NaN detected at iteration {i}. Skipping...")
        skipped += 1
        continue

    mae = mean_absolute_error(actual, predicted)

    current_result = {
        'train_start': df["Date"].iloc[i+time_steps],
        'train_end': df["Date"].iloc[i+initial_train_size+time_steps-1],
        'validation_start': df["Date"].iloc[i+initial_train_size+time_steps],
        'validation_end': df["Date"].iloc[i+initial_train_size+validation_size+time_steps-1],
        'test_date': df["Date"].iloc[i+initial_train_size+validation_size+time_steps],
        'prediction': predicted.flatten()[0],
        'actual': actual.flatten()[0],
        'mae': mae
    }

    print(current_result)
    results.append(current_result)
    counter += 1

results_path = os.path.join(data_dir, "results_lstm_garch_vix_relu.csv")
pd.DataFrame(results).to_csv(results_path, index=False)
print(f"Results saved to {results_path}")
print(f"Total iterations skipped: {skipped}")
