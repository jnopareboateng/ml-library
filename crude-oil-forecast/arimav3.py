
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pmdarima as pm
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.eval_measures import rmse
from math import sqrt
from sklearn.metrics import mean_squared_error,mean_absolute_error, mean_absolute_percentage_error
# from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
# from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")


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
model = pm.auto_arima(train, seasonal=True, stepwise=True, suppress_warnings=True, trace=True, error_action="ignore")


#%%

order = model.order 


#%%

model.plot_diagnostics(figsize=(12, 8)).show()


#%%
# Fit the model with the rraining set and best parameters found by auto_arima 
model = SARIMAX(train, order= order)
model_fit = model.fit()


model_fit.summary(alpha=0.05)



model_fit.summary(alpha=0.10)


# dynamic=False means that forecasts at each point are generated using the full history up to that point
predictions = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)


#%%

# Calculate evaluation metrics
# Mean  Squared Error.
mse = mean_squared_error(test['Price'], predictions)
print(f'MSE: {mse:.3f}')
# Root Mean Squared Error (RMSE)
rmse = sqrt(mean_squared_error(test['Price'], predictions))
print(f'RMSE: {rmse:.3f}')

# Mean Absolute Error (MAE)
mae = mean_absolute_error(test['Price'], predictions)
print(f'MAE: {mae:.3f}')

# Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(test['Price'], predictions)
print(f'MAPE: {mape*100:.3f}%')

#%%
# Plot actual vs predicted with plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=test.index, y=test['Price'], mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=test.index, y=predictions, mode='lines', name='Predicted'))
fig.update_layout(title='Actual vs Predicted Prices', xaxis_title='Date', yaxis_title='Price')
fig.show()




# Forecast for 24 months from 2023-01-01
future_dates = pd.date_range(start='2023-01-01', periods=24, freq='M')

# 95% confidence interval
forecast_obj_95 = model_fit.get_prediction(start=future_dates[0], end=future_dates[-1])
forecast_95 = forecast_obj_95.predicted_mean
conf_int_95 = forecast_obj_95.conf_int(alpha=0.05)

# 90% confidence interval
forecast_obj_90 = model_fit.get_prediction(start=future_dates[0], end=future_dates[-1])
conf_int_90 = forecast_obj_90.conf_int(alpha=0.10)

# Plot actual data, forecast, and confidence intervals
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_95, mode='lines', name='Forecast'))

# 95% confidence interval
fig.add_trace(go.Scatter(x=future_dates, y=conf_int_95.iloc[:, 0], mode='lines', name='Lower 95% CI', line=dict(color='rgba(255,0,0,0.5)')))
fig.add_trace(go.Scatter(x=future_dates, y=conf_int_95.iloc[:, 1], mode='lines', name='Upper 95% CI', line=dict(color='rgba(255,0,0,0.5)'), fill='tonexty'))

# 90% confidence interval
fig.add_trace(go.Scatter(x=future_dates, y=conf_int_90.iloc[:, 0], mode='lines', name='Lower 90% CI', line=dict(color='rgba(0,0,255,0.5)')))
fig.add_trace(go.Scatter(x=future_dates, y=conf_int_90.iloc[:, 1], mode='lines', name='Upper 90% CI', line=dict(color='rgba(0,0,255,0.5)'), fill='tonexty'))

fig.update_layout(title='Brent Crude Oil Price Forecast (with Confidence Intervals)', xaxis_title='Date', yaxis_title='Price')
fig.show()


# print(conf_int)



