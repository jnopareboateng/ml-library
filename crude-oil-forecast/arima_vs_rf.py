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
model = SARIMAX(train, order= order).fit()
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