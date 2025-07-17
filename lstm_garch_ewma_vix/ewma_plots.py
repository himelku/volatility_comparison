import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


os.makedirs("plots", exist_ok=True)
# Paths
data_dir = "data"
ewma_path = os.path.join(data_dir, "results_ewma.csv")
vix_path = os.path.join(data_dir, "vix_15min.csv")

# Load EWMA
ewma = pd.read_csv(ewma_path)
ewma["Date"] = pd.to_datetime(ewma["Date"])
ewma.dropna(subset=["ewma_volatility", "log_returns"], inplace=True)

# --- Plot 1: EWMA vs Rolling Std Dev ---
rolling_std = ewma["log_returns"].rolling(26).std()
plt.figure(figsize=(10, 4))
plt.plot(ewma["Date"], ewma["ewma_volatility"], label="EWMA Volatility")
plt.plot(ewma["Date"], rolling_std, label="Rolling Std Dev", linestyle="--")
plt.title("EWMA vs Rolling Std Dev (26 periods)")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/ewma_vs_rolling_std.png")
plt.close()

# --- Plot 2: EWMA vs Log Returns ---
plt.figure(figsize=(10, 4))
plt.plot(ewma["Date"], ewma["log_returns"], label="Log Returns")
plt.plot(ewma["Date"], ewma["ewma_volatility"], label="EWMA Volatility", alpha=0.7)
plt.title("Log Returns vs EWMA Volatility")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/log_returns_vs_ewma.png")
plt.close()

# --- Plot 3: EWMA Lambda Comparison ---
for lam in [0.94, 0.97, 0.99]:
    ewma_col = f"ewma_{lam}"
    ewma[ewma_col] = ewma["log_returns"].ewm(span=(2 / (1 - lam) - 1)).std()

plt.figure(figsize=(10, 4))
for lam in [0.94, 0.97, 0.99]:
    plt.plot(ewma["Date"], ewma[f"ewma_{lam}"], label=f"λ = {lam}")
plt.title("EWMA Volatility Comparison for Different λ")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/ewma_lambda_comparison.png")
plt.close()

# --- Plot 4: Forecast Error Over Time ---
rolling_std = ewma["log_returns"].rolling(26).std()
forecast_error = np.abs(rolling_std.values - ewma["ewma_volatility"].values)
plt.figure(figsize=(10, 4))
plt.plot(ewma["Date"], forecast_error, label="|Rolling Std - EWMA|")
plt.title("EWMA Forecast Error Over Time")
plt.xlabel("Date")
plt.ylabel("Error")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/ewma_forecast_error.png")
plt.close()

# --- Plot 5: EWMA Regime Shifts ---
threshold = ewma["ewma_volatility"].mean() + ewma["ewma_volatility"].std()
high_vol = ewma["ewma_volatility"] > threshold
plt.figure(figsize=(10, 4))
plt.plot(ewma["Date"], ewma["ewma_volatility"], label="EWMA Volatility")
plt.fill_between(ewma["Date"], 0, ewma["ewma_volatility"], where=high_vol, color='red', alpha=0.3, label="High Volatility Regime")
plt.title("EWMA Volatility Regime Shifts")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/ewma_regime_shifts.png")
plt.close()

# --- Plot 6: Volatility Distribution ---
plt.figure(figsize=(6, 4))
ewma["ewma_volatility"].plot(kind="hist", bins=50, alpha=0.7, density=True)
ewma["ewma_volatility"].plot(kind="kde", label="KDE")
plt.title("EWMA Volatility Distribution")
plt.xlabel("Volatility")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/ewma_distribution.png")
plt.close()

# --- Plot 7: EWMA vs VIX ---
if os.path.exists(vix_path):
    vix = pd.read_csv(vix_path)
    vix["Date"] = pd.to_datetime(vix["timestamp"])
    vix.rename(columns={"close": "Close_vix"}, inplace=True)
    merged = pd.merge(ewma, vix[["Date", "Close_vix"]], on="Date", how="inner")

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(merged["Date"], merged["Close_vix"], color="blue", label="VIX (Close)")
    ax1.set_ylabel("VIX", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(merged["Date"], merged["ewma_volatility"], color="orange", linestyle="--", label="EWMA Volatility")
    ax2.set_ylabel("EWMA", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    plt.title("EWMA vs VIX (Dual Axis)")
    fig.tight_layout()
    plt.savefig("plots/ewma_vs_vix_dual_axis.png")
    plt.close()
else:
    print("⚠️ VIX data not found.")

# --- Plot 8: MACD-style EWMA crossover ---
ewma["ewma_short"] = ewma["log_returns"].ewm(span=10).mean()
ewma["ewma_long"] = ewma["log_returns"].ewm(span=50).mean()
plt.figure(figsize=(10, 4))
plt.plot(ewma["Date"], ewma["ewma_short"], label="EWMA Short (10)")
plt.plot(ewma["Date"], ewma["ewma_long"], label="EWMA Long (50)")
plt.fill_between(ewma["Date"], ewma["ewma_short"], ewma["ewma_long"],
                 where=(ewma["ewma_short"] > ewma["ewma_long"]),
                 color='green', alpha=0.3, label='Bullish Signal')
plt.fill_between(ewma["Date"], ewma["ewma_short"], ewma["ewma_long"],
                 where=(ewma["ewma_short"] < ewma["ewma_long"]),
                 color='red', alpha=0.3, label='Bearish Signal')
plt.title("MACD-style EWMA Crossover")
plt.xlabel("Date")
plt.ylabel("Signal Strength")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/ewma_macd_style_crossover.png")
plt.close()
