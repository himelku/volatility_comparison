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

## 2. Data Collection and Preprocessing
### 2.1 Datasets Used
- SPY 15-minute intraday data (Open, High, Low, Close, Volume) [`fetch_intraday_data.py`]
- VIX 15-minute intraday data [`fetch_vix_15min.py`]
### 2.2 Data Source
- Alpha Vantage API [https://www.alphavantage.co/]
### 2.3 Preprocessing Steps
- Time alignment of SPY and VIX datasets
- Calculation of log returns
- EWMA volatility estimation
- Lagged volatility features
- Train-test split and scaling for LSTM

---

## 3. Model Architectures
### 3.1 GARCH
Implemented using  arch  Python library to produce baseline volatility estimates.
### 3.2 LSTM
Neural network models created using TensorFlow/Keras with tuning of hyperparameters:
- Layers: 1, 2, or 3
- Activation functions: Tanh, ReLU
- Lookback windows: 5, 22, 66 intervals
- Loss functions: MAE, MSE
### 3.3 Hybrid GARCH-LSTM
Combines lagged volatility predictions from GARCH as additional input features for LSTM.
### 3.4 GARCH-LSTM-VIX
Adds VIX values (close) to LSTM input in combination with GARCH outputs.
### 3.5 GARCH-LSTM-EWMA-VIX
Further integrates EWMA volatility and rolling standard deviations for enhanced input diversity.

## 4. Evaluation Metrics
Each model was evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)


## 5. 🗂️ Project Directory Structure

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
### 5.1 Key Scripts
- `garch.py` : Training and prediction for GARCH
- `lstm_intraday.py` : Base LSTM model creation
- `get_best_hyperparameters.py` :
- `lstm_garch_intraday.py` : Hybrid model generation
- `ewma.py` : EWMA volatility calculation
- `lstm_garch_ewma_vix.py` :
- `models_comparison.py` : Evaluation and plotting
- `app.py` : Interactive dashboard for model selection and plot viewing


---

## 6. Visualizations and Interpretation
Using interactive Plotly charts, users can view:
- Predictions vs Actuals for each model
- Sensitivity to lookback windows and activation functions
- EWMA vs Rolling Standard Deviation
- VIX vs EWMA overlays

---

## 7. Interactive Streamlit Dashboard
### Features:
- Sidebar dropdowns for model selection
- Tabs for each model category (GARCH, LSTM, Hybrid, EWMA)
- Dynamic plot generation using Plotly
- Real-time display of MAE and RMSE
### Deployment:
- Supports local and cloud deployment (Streamlit Cloud)
- Uses pre-trained model results stored in  data/

---

## 8. Future Work
- Integrate real-time data ingestion
- Add models like HARCH, HAR, or transformers
- Improve interpretability using SHAP/attention mechanisms
- Compare performance across other ETFs and asset classes
- Extend to multi-asset volatility forecasting

---

## 9. Conclusion
This project builds a robust and extensible pipeline for intraday volatility forecasting. By blending GARCH’s statistical strength with LSTM’s sequence modeling capability and augmenting with market volatility indices (VIX) and technical indicators (EWMA), our hybrid models significantly improve volatility prediction performance.  The  accompanying  dashboard  offers  an  intuitive  and  flexible  interface  for  real-time comparison and visualization


---

## 📝 Notes and Recommendations

- All datasets and result files are directed to `data/` directory.
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

🔍 The Review of Financial Studies, Vol. 19, No. 3 (Autumn, 2006), pp. 871-908. Published by: Oxford University Press. Sponsor: The Society for Financial Studies.

---

## 📅 Project Info.

`Thesis Project – MQIM, University of New Brunswick`\
`Md Mahmudul Hasan`\
`2025`

---

## Appendix
### Tools and Libraries
`Python 3.11` `TensorFlow` `Keras` `pandas` `numpy` `scikit-learn` `arch` `Streamlit` `Plotly` `Matplotlib`


---

## Disclosure of Assistance and External Resources

This project was developed with the support of various open-source tools and AI assistance. Parts of the modeling pipeline and code structure were inspired by publicly available GitHub repositories, which were carefully adapted and referenced where appropriate.

Additionally, OpenAI's ChatGPT (June 2024 version) was used as a supplementary tool to assist in explaining theoretical concepts (such as GARCH and LSTM models), troubleshooting code, drafting documentation, and brainstorming structure. **Overall modeling decisions, final code implementations, interpretations, and written content reflect my own understanding and work.**

*The use of these tools is disclosed in the spirit of transparency and academic integrity, and their role remained supportive rather than generative.*


---
## Acknowledgments
This research project was developed with the support of various open-source tools and repositories. I would like to acknowledge code structures and modeling ideas adapted from publicly available GitHub repositories.

I also wish to acknowledge OpenAI’s ChatGPT, which provided helpful technical explanations and suggestions during model development, troubleshooting code, and documentation. All final decisions regarding modeling choices and interpretations were made independently.

---

© 2025 – University of New Brunswick | Master of Quantitative Investment Management
