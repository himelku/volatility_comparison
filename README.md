# 📊 Intraday Volatility Forecasting with Hybrid GARCH-LSTM-VIX Models  
**A 15-Minute Interval Prediction Framework for SPY ETF using Statistical and Deep Learning Models**

## 🧠 Project Overview

This research project presents a robust intraday volatility forecasting pipeline using 15-minute interval data from the **SPY ETF**. It integrates statistical modeling (**GARCH**), deep learning (**LSTM**), and macroeconomic volatility indicators (**VIX**) to provide accurate short-term volatility predictions.

The objective is to demonstrate how combining econometric techniques with deep learning and market sentiment proxies improves predictive power in high-frequency financial time series.

---

## 🚀 Key Features

- ✅ **GARCH** model to capture mean-reverting conditional volatility
- ✅ **LSTM** neural networks to capture nonlinear temporal dependencies
- ✅ **VIX (Volatility Index)** used as an exogenous feature to enrich contextual information
- ✅ Modular architecture with sensitivity experiments:
  - Lookback windows (5, 22, 66)
  - Loss functions (MAE vs MSE)
  - LSTM architectures (layers, activations)
- ✅ Visualized model comparisons and tabulated evaluation metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)

---

## 🗂️ Project Directory Structure

```
volatility_prediction_project/
│
├── fetch_intraday_data.py
├── fetch_vix_15min.py
├── get_best_hyperparameters.py
├── project.ipynb
├── app.py
|
├── data/
|   ├── results_ewma.csv
|   ├── results_garch_intraday.csv
│   ├── results_lstm_garch_ewma_vix_intraday.csv
|   ├── results_lstm_garch_intraday.csv
|   ├── results_lstm_garch_vix_intraday.csv
|   ├── results_lstm_garch_vix_layer_1_intraday.csv
|   ├── results_lstm_garch_vix_layer_3_intraday.csv
|   ├── results_lstm_garch_vix_lookback_5_intraday.csv
|   ├── results_lstm_garch_vix_lookback_66_intraday.csv
|   ├── results_lstm_garch_vix_mae_loss_intraday.csv
|   ├── results_lstm_garch_vix_pct_change_intraday.csv
|   ├── results_lstm_garch_vix_relu.csv
|   ├── results_lstm_intraday.csv
│   ├── SPY_15min_lstm.csv
│   ├── vix_15min.csv
│   ├── SPY_15min_intraday.csv
│
├── garch/
│   ├── garch.py
│   ├── garch_functions.py
│   ├── garch_parameters_choice.py
│
├── lstm/
│   ├── lstm_functions.py
│   ├── lstm_hyperparameters_tuning_intraday.py
│   ├── lstm_intraday.py
│   ├── prepare_data_lstm_intraday.py
│
├── lstm_garch/
│   ├── lstm_garch_intraday.py
|
├── lstm_garch_ewma_vix/
│   ├── ewma.py
│   ├── ewma_functions.py
│   ├── ewma_plots.py
│   ├── lstm_garch_ewma_vix.py

├── lstm_garch_vix/
│   ├── lstm_garch_vix_intraday.py
|
├── model_tuning_intraday/
│   ├── LSTM_Tuning_intraday
|
├── models_comparison/
│   ├── models_comparison.py
|
├── plots/
│   ├── lstm_garch_intraday.png
    ├── lstm_garch_intraday.png
    ├── ...
|
├── sensitivity/
│   ├── sensitivity_model_function.py
    ├── layer_1.py
│   ├── layer_3.py
│   ├── mae_loss.py
│   ├── pct_change_input.py
│   ├── relu.py
│   ├── sequence_length_5.py
│   ├── sequence_length_66.py
|
├── requirements.txt
└── README.md
```
---

## 💻 How to Run in Google Colab or Locally

<instructions... same as before>

---

## 🧪 Sensitivity Experiments

<same content...>

---

## 📊 Model Evaluation Metrics

<same content...>

---

## 📝 Notes and Recommendations

- All datasets and result files are stored in the `data/` directory.
- Adjust model parameters and sensitivity configs under `sensitivity/` and `lstm/`.
- Use GPU runtime in Colab for faster training (~2–3 mins per model).
- Make sure to run **all variant scripts** before executing the final comparison.

---

## 📚 References

- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. _Journal of Econometrics_
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. _Neural computation_
- GitHub Project Referenced: [Volatility Forecasting](https://github.com/) – Some model logic and architectural design ideas were adapted from this project. All implementation here has been independently modified and documented for academic purposes.

---

## 📅 Project Info

**Course**: Thesis Project – MQIM, University of New Brunswick  
**Contributor**: Md Mahmudul Hasan  
**Year**: 2025  
**License**: MIT

---

© 2025 – University of New Brunswick | Master of Quantitative Investment Management
