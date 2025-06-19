# 📊 Intraday Volatility Forecasting with Hybrid GARCH-LSTM-VIX Models
**A 15-Minute Interval Prediction Framework for SPY ETF using Statistical and Deep Learning Models**

## 🧠 Project Overview

This research project presents a robust intraday volatility forecasting pipeline using 15-minute interval data from the **SPY ETF**. It integrates statistical modeling (**GARCH**), deep learning (**LSTM**), and macroeconomic volatility indicators (**VIX**) to provide accurate short-term volatility predictions.

The objective is to demonstrate how combining econometric techniques with deep learning and market sentiment proxies improves predictive power in high-frequency financial time series.

---

**Literature Review:**

“Building on the findings of Christoffersen and Diebold (2000), and Poon and Granger (2003), I integrate implied volatility into my forecasting framework to enhance accuracy during turbulent market conditions.”

---

**Methodology Justification:**

“Inspired by Blair, Poon, and Taylor (2001), I use 15-minute return data combined with VIX-based exogenous features to forecast SPY volatility.”

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

🔍 Christoffersen, P., & Diebold, F. X. (2000). How relevant is volatility forecasting for financial risk management? Review of Economics and Statistics, 82(1), 12–22.

🔍 Poon, S.-H., & Granger, C. W. J. (2003). Forecasting volatility in financial markets: A review. Journal of Economic Literature, 41(2), 478–539.

🔍 Blair, B. J., Poon, S.-H., & Taylor, S. J. (2001). Forecasting S&P 100 volatility: The incremental information content of implied volatilities and high-frequency index returns. Journal of Econometrics, 105(1), 5–26.

🔍 Taylor, S. J. (2005). Asset price dynamics, volatility, and prediction. Princeton University Press.

🔍 Hansen, P. R., & Lunde, A. (2005). A forecast comparison of volatility models: Does anything beat a GARCH(1,1)? Journal of Applied Econometrics, 20(7), 873–889.

🔍 Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. Econometrica, 59(2), 347–370.

🔍 The Review of Financial Studies, Vol. 19, No. 3 (Autumn, 2006), pp. 871-908
      Published by: Oxford University Press. Sponsor: The Society for Financial Studies. URL: http://www.jstor.org/stable/3844016

---

## 📅 Project Info.

**Course**: Thesis Project – MQIM, University of New Brunswick\
**Contributor**: Md Mahmudul Hasan\
**Year**: 2025\
**License**: MIT

---

## Disclosure of Assistance and External Resources

This project was developed with the support of various open-source tools and AI assistance. Parts of the modeling pipeline and code structure were inspired by publicly available GitHub repositories, which were carefully adapted and referenced where appropriate.

Additionally, OpenAI's ChatGPT (June 2024 version) was used as a supplementary tool to assist in explaining theoretical concepts (such as GARCH and LSTM models), troubleshooting code, drafting documentation, and brainstorming presentation structure. All modeling decisions, final code implementations, interpretations, and written content reflect my own understanding and work, developed under the guidance of my academic supervisor.

The use of these tools is disclosed in the spirit of transparency and academic integrity, and their role remained supportive rather than generative.


---
## Acknowledgments
This research project was developed with the support of various open-source tools and repositories. I would like to acknowledge code structures and modeling ideas adapted from public GitHub repositories.

I also wish to acknowledge OpenAI’s ChatGPT, which provided helpful technical explanations and suggestions during model development, coding, and documentation. All final decisions regarding modeling choices and interpretations were made independently.

---

© 2025 – University of New Brunswick | Master of Quantitative Investment Management
