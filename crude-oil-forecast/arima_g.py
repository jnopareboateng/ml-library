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
data = pd.read_csv('Modified data.csv', parse_dates=True, index_col=[0])

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
# Fit the SARIMA model on the differenced training data
model = SARIMAX(endog=differenced_data, order=model.order, seasonal_order=model.seasonal_order, freq="MS")
results = model.fit(disp=0)  # Suppress convergence output
print(results.summary())

#%%
# Get forecasts and confidence intervals on validation data
forecast = results.get_forecast(steps=24)
forecast_summary = forecast.summary_frame(alpha=0.05)

#%%
# Inverse transform predictions if differencing was performed
# if n_diffs > 0:
#     predictions = results.predict(steps=24) + differenced_data.iloc[-n_diffs:].values

# Inverse transform predictions if differencing was performed
if n_diffs > 0:
    predictions = pd.Series(results.predict(steps=24), index=differenced_data.index[-24:])
    predictions = predictions.cumsum() + data.iloc[-n_diffs]
#%%
# Evaluate model performance on validation data (e.g., MSE, MAPE)

# Plot forecast with confidence intervals
# (Similar to previous code, but using validation_data and predicted values)

# ... (plotting and interpretation of forecast and confidence intervals)


# %%
