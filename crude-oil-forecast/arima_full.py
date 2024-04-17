# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import warnings
import pmdarima as pm 
import math
from pmdarima import auto_arima
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from common.preprocessor import load_data
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")
%matplotlib inline

# %%
# load data from the preprocessor and set index to date column
data = pd.read_csv('Modified data.csv', parse_dates=True, index_col=[0])

# %%
data.head() # display the first 5 rows of the data

# %%
data.describe() # display the summary statistics of the data

# %%

# visualize the data

px.line(data, x=data.index, y=data['Price'], title="Brent Crude Oil Prices from 2002 -2022")

# %%
# Visualize the trend, seasonal component, and residuals
decomposition = seasonal_decompose(data["Price"], model="additive")
decomposition.plot()
plt.show()  

# %%
# Check for stationarity with ADF test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("Data is likely stationary.")
    else:
        print("Data may be non-stationary. Consider differencing.")

print("""Testing stationarity of data:""")
adf_test(data)


# %%
# shapiro wilk test
from scipy.stats import shapiro
stat, p = shapiro(data)
print('Statistics=%.4f, p=%.7f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# %%
# Identify number of differences required (if necessary)
n_diffs = pm.arima.ndiffs(data['Price'], test='adf')
print(f"\nNumber of differences required for scaled training data: {n_diffs}")


# %%
# Perform differencing if required
if n_diffs > 0:
    differenced_data = data.diff(n_diffs).dropna()
else:
    differenced_data = data.copy()

# %%
# plot differenced data
differenced_data.plot()

# %%
# check seasonal differencing
from pmdarima.arima.utils import nsdiffs

nsdiff= nsdiffs(data['Price'], m=12, test='ch')
print(f"Seasonal differences required: {nsdiff}")

# %%
# Visualize the trend, seasonal component, and residual after differencing
decomposition = seasonal_decompose(differenced_data["Price"], model="additive")  # "Price" is likely your column name for oil prices
decomposition.plot()
plt.show()  


# %%
# ACF and PACF plots (optional) on differenced data)
plot_acf(differenced_data['Price'], title='ACF Plot')
plt.show()
plot_pacf(differenced_data['Price'], title='PACF Plot ')
plt.show()

# %%
# visually inspect the seasonality of the data
df_2002 = data['2002']
df_2003 = data['2003']
df_2004 = data['2004']
df_2005 = data['2005']
df_2006 = data['2006']
df_2007 = data['2007']
# Create subplot figure
fig = make_subplots(rows=6, cols=1)

# Add traces
fig.add_trace(go.Scatter(x=df_2002.index, y=df_2002['Price'], name='Price in 2002'), row=1, col=1)
fig.add_trace(go.Scatter(x=df_2003.index, y=df_2003['Price'], name='Price in 2003'), row=2, col=1)
fig.add_trace(go.Scatter(x=df_2004.index, y=df_2004['Price'], name='Price in 2004'), row=3, col=1)
fig.add_trace(go.Scatter(x=df_2005.index, y=df_2005['Price'], name='Price in 2005'), row=4, col=1)
fig.add_trace(go.Scatter(x=df_2006.index, y=df_2006['Price'], name='Price in 2006'), row=5, col=1)
fig.add_trace(go.Scatter(x=df_2007.index, y=df_2007['Price'], name='Price in 2007'), row=6, col=1)


# Update xaxis properties
fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_xaxes(title_text="Date", row=3, col=1)
fig.update_xaxes(title_text="Date", row=4, col=1)
fig.update_xaxes(title_text="Date", row=5, col=1)
fig.update_xaxes(title_text="Date", row=6, col=1)




# Update yaxis properties
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Price", row=2, col=1)
fig.update_yaxes(title_text="Price", row=3, col=1)
fig.update_yaxes(title_text="Price", row=4, col=1)
fig.update_yaxes(title_text="Price", row=5, col=1)
fig.update_yaxes(title_text="Price", row=6, col=1)


# Update layout
fig.update_layout(height=1000, title_text="Price from 2002 to 2007")

fig.show()

# %%
# Check for stationarity on differenced data with ADF test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("Data is likely stationary.")
    else:
        print("Data may be non-stationary. Consider differencing.")

print("Testing stationarity of scaled training data:")
adf_test(differenced_data['Price'])

# %%
# ACF and PACF plots on differenced data
plot_acf(differenced_data['Price'], title='ACF Plot')
plt.show()
plot_pacf(differenced_data['Price'], title='PACF Plot')
plt.show()

# %%
# Use auto_arima to find best parameters
model = auto_arima(differenced_data['Price'], trace=True,)
print(f"\nAuto ARIMA identified parameters: {model.order}, {model.seasonal_order}")
# decide tradeoff between time and aic

# %%
# show the model order and seasonal order
print(f'model order: {model.order}, \nmodel seasonal order: {model.seasonal_order}')

# %%
# plot the residuals of the model
model.plot_diagnostics(figsize=(12, 8))
plt.show()

# %%
# Fit the SARIMA model on the differenced training data
model = SARIMAX(endog=differenced_data, order=model.order, seasonal_order=model.seasonal_order, freq="MS")
results = model.fit(disp=0)  # Suppress convergence output
print(results.summary())

# %%
# Get confidence intervals of forecasts
forecast = results.get_forecast(steps=24)
forecast_summary = forecast.summary_frame(alpha=0.05)
forecast_summary.head()

# %%
# Assuming 'data' is a DataFrame with a 'Price' column
history = data['Price']
HORIZON = 24

# Assuming 'model' is a previously defined SARIMAX model
order = model.order
seasonal_order = model.seasonal_order

# Fit the model and forecast the next 24 steps
model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
model_fit = model.fit()
predictions = model_fit.forecast(steps=HORIZON)

# Create a date range for the forecast period
forecast_period = pd.date_range(start=history.index[-1], periods=HORIZON+1, freq='MS')[1:]

# Convert predictions to a pandas Series with the forecast period as index
predictions = pd.Series(predictions, index=forecast_period)

# Get confidence intervals of forecasts at 90% and 95%
forecast_90 = model_fit.get_forecast(steps=HORIZON)
forecast_summary_90 = forecast_90.summary_frame(alpha=0.10)

forecast_95 = model_fit.get_forecast(steps=HORIZON)
forecast_summary_95 = forecast_95.summary_frame(alpha=0.05)

# Plotting the historical data
plt.plot(history.index, history, label='Historical Data')

# Plotting the forecasted values
plt.plot(predictions.index, predictions, color='red', label='Forecasted Values')

# Plotting the 90% confidence intervals
plt.fill_between(forecast_summary_90.index,
                 forecast_summary_90['mean_ci_lower'],
                 forecast_summary_90['mean_ci_upper'], color='pink', alpha=0.3, label='90% Confidence Interval')

# Plotting the 95% confidence intervals
plt.fill_between(forecast_summary_95.index,
                 forecast_summary_95['mean_ci_lower'],
                 forecast_summary_95['mean_ci_upper'], color='blue', alpha=0.2, label='95% Confidence Interval')

plt.title('Forecast with Confidence Intervals')
plt.legend()
plt.show()


# %%
# Create a new figure
fig = go.Figure()

# Add historical data to the figure
fig.add_trace(go.Scatter(x=history.index, y=history, mode='lines', name='Historical Data'))

# Add forecasted values to the figure
fig.add_trace(go.Scatter(x=predictions.index, y=predictions, mode='lines', name='Forecasted Values'))

# Add 90% confidence interval to the figure
fig.add_trace(go.Scatter(x=forecast_summary_90.index, y=forecast_summary_90['mean_ci_upper'], mode='lines', name='90% Confidence Interval', line=dict(width=0)))
fig.add_trace(go.Scatter(x=forecast_summary_90.index, y=forecast_summary_90['mean_ci_lower'], mode='lines', name='90% Confidence Interval', line=dict(width=0), fill='tonexty'))

# Add 95% confidence interval to the figure
fig.add_trace(go.Scatter(x=forecast_summary_95.index, y=forecast_summary_95['mean_ci_upper'], mode='lines', name='95% Confidence Interval', line=dict(width=0)))
fig.add_trace(go.Scatter(x=forecast_summary_95.index, y=forecast_summary_95['mean_ci_lower'], mode='lines', name='95% Confidence Interval', line=dict(width=0), fill='tonexty'))

# Set the title of the figure
fig.update_layout(title='Forecast with Confidence Intervals')

# Show the figure
fig.show()

#%%
# FIXME: FIGURE OUT THE ERROR METRICS FOR THE ANALYSIS 
# calculate the errors using mean absolute error and mean absolute percentage error
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
mae = mean_absolute_error(data, predictions)
mape = mean_absolute_percentage_error(data, predictions)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Absolute Percentage Error: {mape}")

# %%
