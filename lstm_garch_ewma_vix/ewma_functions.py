import numpy as np
import pandas as pd

def compute_ewma_volatility(log_returns, lambda_=0.94):
    ewma_vol = [np.nan]
    for i in range(1, len(log_returns)):
        prev_vol_sq = ewma_vol[-1]**2 if not np.isnan(ewma_vol[-1]) else np.nan
        if np.isnan(prev_vol_sq):
            prev_vol_sq = np.mean(log_returns[:i]**2)
        sigma2 = (1 - lambda_) * log_returns[i - 1] ** 2 + lambda_ * prev_vol_sq
        ewma_vol.append(np.sqrt(sigma2))
    return pd.Series(ewma_vol, index=log_returns.index)