#%%
# import necessary libraries for arima and XG Boost
from tracemalloc import start
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
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
# visualize the train and test data
# import plotly.graph_objects as go

# Create a trace for the train data
train_trace = go.Scatter(
    x = data[(data.index < test_start_date) & (data.index >= train_start_date)].index,
    y = data[(data.index < test_start_date) & (data.index >= train_start_date)]['Price'],
    mode = 'lines',
    name = 'train'
)

# Create a trace for the test data
test_trace = go.Scatter(
    x = data[test_start_date:].index,
    y = data[test_start_date:]['Price'],
    mode = 'lines',
    name = 'test'
)

# Create the layout
layout = go.Layout(
    title = 'Train and Test Data',
    xaxis = dict(title = 'Timestamp'),
    yaxis = dict(title = 'Price')
)

# Create the figure and add the traces
fig = go.Figure(data=[train_trace, test_trace], layout=layout)

# Show the figure
fig.show()

#%%
# set the train and test data and print the dimensions of it
train = data.copy()[(data.index >= train_start_date) & (data.index < test_start_date)][['Price']]
test = data.copy()[data.index >= test_start_date][['Price']]

print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)
#%%
series = pd.Series(data=train['Price'].to_numpy(), index =train.index)
# series

#%%
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

adf_test(series)

#%%
# px.line(series[0:100])
#%%
# Decompose time series
result = seasonal_decompose(series, model='additive', period=1)
result.plot()
plt.show()


#%%
# Acf Plot
# plot_acf(series).show()

#%%
# plot_pacf(series).show()

#%%
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
plot_pacf(diff).show()

#%%
# Find order of MA term Q
plot_acf(diff).show()

#%%
# use auto_arima to find best parameters
model = pm.auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True, trace=True, error_action="ignore")

#%%
order = model.order 


#%%
model.plot_diagnostics(figsize=(12, 8)).show()

#%%
model = SARIMAX(train, order= order)
model_fit = model.fit()
# %%
predictions = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)

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
print('MAPE:', mape, '%')


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
# Random Forest Classifier
# Split the data into features and target
X_train, y_train = np.array(train.index).reshape(-1,1), train.values
X_test, y_test = np.array(test.index).reshape(-1,1), test.values
#%%
# Initialize Random forest regressor
rf = RandomForestRegressor()
#%%

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Perform randomized search CV
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)
#%%

# Fit the model with the best parameters
rf_best = RandomForestRegressor(n_estimators=rf_random.best_params_['n_estimators'], 
                                max_depth=rf_random.best_params_['max_depth'], 
                                min_samples_split=rf_random.best_params_['min_samples_split'], 
                                min_samples_leaf=rf_random.best_params_['min_samples_leaf'], 
                                bootstrap=rf_random.best_params_['bootstrap'])
rf_best.fit(X_train, y_train)
#%%

# Predict on test data
rf_predictions = rf_best.predict(X_test)
#%%

# Calculate evaluation metrics
rmse_rf = sqrt(mean_squared_error(y_test, rf_predictions))
mae_rf = mean_absolute_error(y_test, rf_predictions)
mape_rf = mean_absolute_percentage_error(y_test, rf_predictions)

print('Random Forest RMSE:', rmse_rf)
print('Random Forest MAE:', mae_rf)
print('Random Forest MAPE:', mape_rf, '%')
#%%

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(test.index, y_test, marker='o', label='Actual')
plt.plot(test.index, rf_predictions, marker='x', label='Predicted')
plt.title('Actual vs Predicted Prices - Random Forest')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

#%%

# Forecast for 48 months from 2022-01-01 to 2024-12-31
future_dates_rf = pd.date_range(start='2022-01-01', end='2024-12-31', freq='M')

#%%

# Create a dummy X input with the required shape for the future dates
X_future_rf = np.array(future_dates_rf).reshape(-1,1)

# Predict on the future dates
rf_forecast = rf_best.predict(X_future_rf)

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Price'], marker='o', label='Actual')
plt.plot(future_dates_rf, rf_forecast, marker='x', label='Forecast')
plt.title('Brent Crude Oil Price Forecast (Random Forest)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


# %%
