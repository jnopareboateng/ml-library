#%%
# import necessary libraries for arima and XG Boost
# from tracemalloc import start
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pmdarima as pm
import logging as log
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_squared_error,mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
# from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

#%%

# Configure logging to print to standard output
log.basicConfig(level=log.INFO, format='%(message)s')


#%%
# Load data from csv file
data = pd.read_csv('Modified_Data.csv', index_col=[0], parse_dates=True)
data.head()


#%%
# Plot data
fig = px.line(data, x=data.index, y='Price', title='Close Price')
fig.show()

#%%
# set the train and test data with start dates
train_start_date = '2002-01-01'
test_start_date = '2019-01-01'


#%%
# set the train and test data and print the dimensions of it
train = data.copy()[(data.index >= train_start_date) & (data.index < test_start_date)][['Price']]
test = data.copy()[data.index >= test_start_date][['Price']]

print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)
#%%
# Plot train and test splits
fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index, y=train['Price'], mode='lines', name='Train'))
fig.add_trace(go.Scatter(x=test.index, y=test['Price'], mode='lines', name='Test'))
fig.update_layout(title='Train and Test Split', xaxis_title='Date', yaxis_title='Price')
fig.show()

#%%
# Convert data to series
series = pd.Series(data=train['Price'].to_numpy(), index =train.index)


#%%
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])

adf_test(series)

#%%
# Decompose time series into trend, seasonal, and residual components
result = seasonal_decompose(series, model='additive', period=1)
result.plot()
plt.show()



#%%
# Differencing to make the series stationary and plot the differenced series
diff = series.diff(periods=1).dropna()
diff_2 = diff.diff(periods=1).dropna()

# plot and add legend in top right corner with plotly subplots
fig = make_subplots(rows=3, cols=1)
fig.add_trace(go.Scatter(x=diff.index, y=diff, name='1st Difference'), row=1, col=1)
fig.add_trace(go.Scatter(x=diff_2.index, y=diff_2, name='2nd Difference'), row=2, col=1)
fig.add_trace(go.Scatter(x=series.index, y=series, name='Original'), row=3, col=1)
fig.show()


#%%
# number of  differencing for stationary series with ndiffs
from pmdarima.arima.utils import ndiffs
y = series
# Adf Test
print('ADF:', ndiffs(y, test='adf'))
# KPSS test
print('KPSS:', ndiffs(y, test='kpss'))
# PP test:
print('PP:', ndiffs(y, test='pp'))

#%%
# Order of auto regressive term P
plot_pacf(diff, lags =48).show()

#%%
# Find order of MA term Q
plot_acf(diff, lags=48).show()

#%%
# use auto_arima to find best parameters
model = pm.auto_arima(diff, seasonal=True, stepwise=True, suppress_warnings=True, trace=True, error_action="ignore")

#%%
order = model.order 


#%%
model.plot_diagnostics(figsize=(12, 8)).show()

#%%
model = SARIMAX(train, order= order)
model_fit = model.fit()
# %%
predictions = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False) # dynamic=False means that forecasts at each point are generated using the full history up to that point

#%%
# Calculate evaluation metrics
# Root Mean Squared Error (RMSE)
rmse = sqrt(mean_squared_error(test['Price'], predictions))
print('RMSE:', rmse)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(test['Price'], predictions)
print('MAE:', mae)

# Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(test['Price'], predictions)
print('MAPE:', mape*100, '%')


# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(test.index, test['Price'], marker='o', label='Actual')
plt.plot(test.index, predictions, marker='x', label='Predicted')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
# %%
# Forecast for 24 months from 2023-01-01
future_dates = pd.date_range(start='2023-01-01', periods=24, freq='M')
forecast_obj = model_fit.get_prediction(start=future_dates[0], end=future_dates[-1])
forecast = forecast_obj.predicted_mean
conf_int = forecast_obj.conf_int(alpha=0.05)  # 95% confidence interval

# Plot actual data, forecast, and confidence intervals
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Price'], marker='o', label='Actual')
plt.plot(future_dates, forecast, marker='x', label='Forecast')
plt.fill_between(
    future_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], alpha=0.5, color='b', label='Confidence Interval'
)
plt.title('Brent Crude Oil Price Forecast (with Confidence Intervals)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
# %%
# Plot actual data, forecast, and confidence intervals
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Forecast'))
fig.add_trace(go.Scatter(x=future_dates, y=conf_int.iloc[:, 0], mode='lines', name='Lower CI'))
fig.add_trace(go.Scatter(x=future_dates, y=conf_int.iloc[:, 1], mode='lines', name='Upper CI'))
fig.update_layout(title='Brent Crude Oil Price Forecast (with Confidence Intervals)', xaxis_title='Date', yaxis_title='Price')
fig.show()
# %%