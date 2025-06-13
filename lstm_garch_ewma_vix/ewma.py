import pandas as pd
import numpy as np
import os
from ewma_functions import compute_ewma_volatility

data_path = os.path.join("data", "SPY_15min_intraday.csv")
df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["timestamp"])

df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
df["ewma_volatility"] = compute_ewma_volatility(df["log_returns"], lambda_=0.94)

df.dropna(subset=["ewma_volatility"], inplace=True)
df[["Date", "log_returns", "ewma_volatility"]].to_csv("data/results_ewma.csv", index=False)
print("✅ EWMA volatility saved to data/results_ewma.csv")