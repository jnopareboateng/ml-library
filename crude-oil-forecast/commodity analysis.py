#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima, ndiffs, nsdiffs
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import shapiro

#%% Read data
data = pd.read_csv('Modified Data.csv')  # replace 'your_file.csv' with your file path
data_ts = data['Price']  # replace 'Price' with your column name

#%% Plot data
data_ts.plot()
plt.show()

#%% Decompose data
result = seasonal_decompose(data_ts, model='additive', period=12)
result.plot()
plt.show()

#%% Shapiro-Wilk test
print(shapiro(data_ts))

#%% Differencing
n_diffs = ndiffs(data_ts, test='adf')
print(f"\nNumber of differences required: {n_diffs}")

nsdiff= nsdiffs(data_ts, m=12, test='ch')
print(f"\nSeasonal differences required: {nsdiff}")

# %% Perform differencing if required
if n_diffs > 0:
    differenced_data = data_ts.diff(n_diffs).dropna()
else:
    differenced_data = data_ts.copy()


#%% ACF and PACF
plot_acf(data_ts)
plot_pacf(data_ts)
plt.show()

#%% ADF test
result = adfuller(data_ts)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

#%% ARIMA model
model = auto_arima(differenced_data, trace=True)
model.fit(differenced_data)

#%% Forecast
forecast = model.forecast(n_periods=24)
plt.plot(forecast)
plt.show()

# Accuracy (replace 'test' with your test data)
# from sklearn.metrics import mean_squared_error
# print(np.sqrt(mean_squared_error(test, forecast)))

#%% Histogram
plt.hist(data_ts, color='blue')
plt.show()

#%% Density plot
data_ts.plot(kind='kde', color='green')
plt.show()

#%% Linear regression (replace 'time' with your time data)
slope, intercept = np.polyfit(data['time'], data_ts, 1)  # replace 'time' with your time column
print(f'Slope: {slope}, Intercept: {intercept}')

# %%
