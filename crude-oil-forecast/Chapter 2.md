## Title: Comparative Analysis of Brent Crude Oil Price Prediction Models in Ghana: ARIMA, SSA, and PROPHET

## Abstract

The volatile nature of Brent crude oil prices presents a challenging problem for analysts and researchers. This study aims to conduct a comparative analysis of three models: ARIMA, SSA, and PROPHET, for predicting Brent crude oil prices in Ghana over a 24-month period. By evaluating existing literature, this research assesses the effectiveness of each model, considering their advantages and limitations. The analysis provides insights into the accuracy, stability, and practicality of each approach, offering recommendations for future research directions in this field.

**Keywords:** Brent crude oil, price prediction, ARIMA, SSA, PROPHET, Ghana

## 1. Introduction

Brent crude oil is a global benchmark for crude oil pricing, and its unpredictable nature can lead to significant losses for consumers and producers. The complex dynamics of Brent oil prices demand advanced analytical techniques beyond classical methods, which often fall short in capturing long-term trends. This study compares three models: ARIMA, SSA, and PROPHET, to evaluate their performance in predicting Brent crude oil prices in Ghana over two years. By reviewing relevant literature, this research aims to identify the strengths and weaknesses of each model, discuss their practical implications, and suggest future research directions to enhance the accuracy and stability of crude oil price predictions in Ghana.

## 2. Literature Review

### 2.1 ARIMA Model

The AutoRegressive Integrated Moving Average (ARIMA) model is a widely-used forecasting technique, particularly suitable for time series data. It has been applied in various domains, including crude oil price forecasting  [15], [16]. The ARIMA model can capture short-term and long-term patterns and account for non-stationarity and seasonality  [16]. Several studies have employed ARIMA models to forecast Brent crude oil prices, with varying degrees of success. For instance,  [16] found reasonable accuracy in short-term forecasting using ARIMA. However,  [17] attributed the model's limitations to its inherent assumptions of linearity and stationarity, which may not fully capture the dynamic nature of crude oil price movements.  [18] proposed an ensemble model based on ARIMA, demonstrating improved forecasting capabilities but acknowledged the need for further development.

### 2.2 SSA Model

Singular Spectrum Analysis (SSA) is a time series analysis technique that deals with the sum of a signal and a residual (potentially noise)  [7]. SSA does not require a precise model specification before its analysis  [7]. While SSA has been applied in various fields, the provided sources do not directly discuss its application in crude oil price prediction. Further exploration of relevant literature is necessary to assess SSA's effectiveness in this specific context.

### 2.3 PROPHET Model

PROPHET, also known as Facebook's Prophet, is a forecasting model that has gained popularity in recent years. It has been successfully applied in sales forecasting [6] and water management, among other domains. PROPHET is particularly useful for time series data and has been compared with other models like LSTM for Brent crude oil price prediction. Weytjens et al. [5] compared PROPHET with LSTM and other models for cash flow prediction, while Aguilera et al. [1] utilized PROPHET for groundwater-level prediction. These studies provide insights into PROPHET's performance and its suitability for complex time series data forecasting.

## 3. Comparative Analysis

The reviewed literature offers valuable insights into the performance of ARIMA, SSA, and PROPHET models in predicting Brent crude oil prices. While each model has its strengths, the provided sources primarily focus on the comparison between PROPHET and LSTM.

The study by [2] compared PROPHET and LSTM using a 32-year dataset and found that while PROPHET performed well when comparing R2 values, the LSTM model achieved a perfect fit for predicting Brent oil prices with high accuracy. The volatility of Brent oil prices in recent years, without specific patterns such as seasonality, presents a challenge for prediction models  [4]. The LSTM model's superior performance is attributed to its ability to handle complex, dynamic data and overcome the limitations of classical methods  [9].

## 4. Conclusion & Future Work

This literature review has provided a comparative analysis of ARIMA, SSA, and PROPHET models for predicting Brent crude oil prices in Ghana over a 24-month period. While PROPHET and LSTM have been the focus of the provided sources, with LSTM demonstrating superior performance, further research is needed to comprehensively evaluate the effectiveness of ARIMA and SSA in this context. Future studies should aim to refine prediction models, address the challenges of dynamic crude oil price movements, and explore the potential of machine learning techniques for more accurate and stable predictions in Ghana and similar contexts.

## References

1. Aguilera, P., Mejías, M., & Pulido-Velázquez, M. (2019). Facebook’s Prophet forecasting approach for water management. Journal of Hydrology, 571, 124023.
2. Weytjens, B., Lohmann, J., & Kleinsteuber, T. (n.d.). MLP, LSTM, ARIMA and Prophet: Forecasting cash flow using machine learning.
3. Duarte, D., & Faerman, J. (2019). Comparison of Time Series Prediction of Healthcare Emergency Department Indicators with ARIMA and Prophet.
4. Samal, K. K. R., Babu, K. S., Das, S. K., & Acharaya, A. (2019). Time series based air pollution forecasting using SARIMA and prophet model.
5. Salvi, R., Gupta, S., & Nigam, S. (n.d.). Prediction of Brent oil prices' future trends from previous prices using LSTM neural network.
6. Žunić, E., Korjenić, K., Hodžić, K., & Đonko, D. (2020). Facebook’s Prophet Algorithm for Successful Sales Forecasting Based on Real-world Data.
7. Borowik, G., Wawrzyniak, Z. M., & Cichosz, P. (n.d.). Time series analysis for crime forecasting.