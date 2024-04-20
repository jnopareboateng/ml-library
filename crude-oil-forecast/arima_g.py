#%%
import pandas as pd
import numpy as np
import pmdarima as pm
import warnings
from pmdarima import auto_arima
from pmdarima.arima.utils import nsdiffs
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Suppress warnings
pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")
%matplotlib inline
#%%
# Load data
# data = pd.read_csv('Modified data.csv', parse_dates=True, index_col=[0])

# set data this url link ` https://github.com/jnopareboateng/ml-library/blob/master/crude-oil-forecast/Modified%20Data.csv`
data = pd.read_csv('https://raw.githubusercontent.com/jnopareboateng/ml-library/master/crude-oil-forecast/Modified_Data.csv', parse_dates=True, index_col=[0])

# Check for missing values
if data.isnull().sum().any():
    print("Missing values detected. Consider handling them (e.g., drop, interpolation).")

# Explore data (descriptive statistics, visualizations)
print(data.describe())
px.line(data, x=data.index, y=data['Price'], title="Brent Crude Oil Prices")
plt.show()

#%%
# Check stationarity with ADF test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("Data is likely stationary.")
    else:
        print("Data may be non-stationary. Consider differencing.")

print("""Testing stationarity of data:""")
adf_test(data['Price'])

#%%
# Visualize trend, seasonality, and residuals using seasonal decomposition
decomposition = seasonal_decompose(data["Price"], model="additive")
decomposition.plot()
plt.show()
#%%
# Identify required number of differences (if necessary)
n_diffs = pm.arima.ndiffs(data['Price'], test='adf')
print(f"\nNumber of differences required: {n_diffs}")
#%%
# Perform differencing if required
if n_diffs > 0:
    differenced_data = data.diff(n_diffs).dropna()
else:
    differenced_data = data.copy()

#%%
# Check for seasonal differencing

nsdiff = nsdiffs(differenced_data['Price'], m=12, test='ch')
print(f"Seasonal differences required: {nsdiff}")
#%%
# Visualize trend, seasonality, and residuals after differencing
decomposition = seasonal_decompose(differenced_data["Price"], model="additive")
decomposition.plot()
plt.show()

#%%
# ACF and PACF plots on differenced data
plot_acf(differenced_data['Price'], title='ACF Plot')
plt.show()
plot_pacf(differenced_data['Price'], title='PACF Plot')
plt.show()

# Split data into training and validation sets (consider using libraries like scikit-learn)
# train_data, validation_data = ... (implementation depends on chosen library)

#%%
# Use auto_arima to find best parameters on training data
model = auto_arima(differenced_data['Price'], trace=True)
print(f"\nAuto ARIMA identified parameters: {model.order}, {model.seasonal_order}")

# Consider hyperparameter tuning for more robustness (e.g., grid search, randomized search)
#%%
# # Fit the SARIMA model on the differenced training data
# model = SARIMAX(endog=differenced_data, order=model.order, seasonal_order=model.seasonal_order, freq="MS")
# results = model.fit(disp=0)  # Suppress convergence output
# print(results.summary())

# #%%
# # Inverse transform predictions if differencing was performed
# if n_diffs > 0:
#     predictions = pd.Series(results.predict(steps=24), index=differenced_data.index[-24:])
#     predictions = predictions.cumsum() + data.iloc[-n_diffs]
# #%%
# # Evaluate model performance on validation data (e.g., MSE, MAPE)


# # Plot forecast with confidence intervals
# # (Similar to previous code, but using validation_data and predicted values)
# # Get forecasts and confidence intervals on validation data
# # Get confidence intervals of forecasts at 90% and 95%
# HORIZON = 24
# history = data['Price']

# # Create a date range for the forecast period
# forecast_period = pd.date_range(start=history.index[-1], periods=HORIZON+1, freq='MS')[1:]

# # Convert predictions to a pandas Series with the forecast period as index
# predictions = pd.Series(predictions, index=forecast_period)

# forecast_90 = results.get_forecast(steps=HORIZON)
# forecast_summary_90 = forecast_90.summary_frame(alpha=0.10)

# forecast_95 = results.get_forecast(steps=HORIZON)
# forecast_summary_95 = forecast_95.summary_frame(alpha=0.05)

# # Plotting the historical data
# plt.plot(history.index, history, label='Historical Data')

# # Plotting the forecasted values
# plt.plot(predictions.index, predictions, color='red', label='Forecasted Values')

# # Plotting the 90% confidence intervals
# plt.fill_between(forecast_summary_90.index,
#                  forecast_summary_90['mean_ci_lower'],
#                  forecast_summary_90['mean_ci_upper'], color='pink', alpha=0.3, label='90% Confidence Interval')

# # Plotting the 95% confidence intervals
# plt.fill_between(forecast_summary_95.index,
#                  forecast_summary_95['mean_ci_lower'],
#                  forecast_summary_95['mean_ci_upper'], color='blue', alpha=0.2, label='95% Confidence Interval')

# plt.title('Forecast with Confidence Intervals')
# plt.legend()
# plt.show()

# ... (plotting and interpretation of forecast and confidence intervals)


# %%
# Split data into training and validation sets
train_size = int(len(differenced_data) * 0.8)
train, test = differenced_data[0:train_size], differenced_data[train_size:]

# Fit the SARIMA model on the differenced training data
model = SARIMAX(endog=train, order=model.order, seasonal_order=model.seasonal_order, freq="MS")
results = model.fit(disp=0)  # Suppress convergence output

# Make predictions
predictions = results.predict(start=len(train), end=len(train) + len(test) - 1)

# Inverse transform predictions if differencing was performed
if n_diffs > 0:
    predictions = np.r_[train.iloc[0], predictions].cumsum()[1:]

# Evaluate model performance on validation data
mse = mean_squared_error(test, predictions)
mape = mean_absolute_percentage_error(test, predictions)

print(f"MSE: {mse}")
print(f"MAPE: {mape}")
