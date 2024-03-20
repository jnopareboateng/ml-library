import pandas as pd
import numpy as np
import warnings
import pmdarima as pm
import time
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from common.preprocessor import load_data, mape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# %%
import matplotlib.pyplot as plt
# %matplotlib inline
pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")

# %%
start_time = time.time()
data = pd.read_csv('Modified Data.csv', parse_dates=True, index_col=[0])

# %%
data.head()

# %%
data.plot(y='Price', title='Commodity Prices Monthly', figsize=(12, 6))
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.show()

# %%
# Create training and testing datasets
train, test = train_test_split(data, test_size=0.2, shuffle=False)

# %%
train.shape, test.shape

# %%
# Visualize the training and testing datasets
plt.figure(figsize=(12, 6))
plt.plot(train, label='Training')
plt.plot(test, label='Testing')
plt.title('Commodity Prices Monthly')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend()
plt.show()

# %%
# Prepare data for training
scaler = MinMaxScaler()
scaled_train = train.copy()
scaled_test = test.copy()
scaled_train['Price'] = scaler.fit_transform(scaled_train[['Price']])

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

print("Testing stationarity of scaled training data:")
adf_test(scaled_train['Price'])

print("\nTesting stationarity of scaled test data:")
adf_test(scaled_test['Price'])

# %%
# Identify number of differences required (if necessary)
n_diffs = pm.arima.ndiffs(scaled_train['Price'], test='adf')
print(f"\nNumber of differences required for scaled training data: {n_diffs}")

n_diffs = pm.arima.ndiffs(scaled_test['Price'], test='adf')
print(f"\nNumber of differences required for scaled test data: {n_diffs}")

# %%
# Perform differencing if required
if n_diffs > 0:
    differenced_train = scaled_train.diff(n_diffs).dropna()
    differenced_test = scaled_test.diff(n_diffs).dropna()
else:
    differenced_train = scaled_train.copy()
    differenced_test = scaled_test.copy()

# %%
# ACF and PACF plots (optional)
plot_acf(differenced_train['Price'], lags=20, title='ACF for AR(1) process')
plt.show()
plot_pacf(differenced_train['Price'], lags=20, title='PACF for AR(1) process')
plt.show()

# %%
# Use auto_arima to find best parameters
model = auto_arima(differenced_train['Price'], start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                  start_P=0, seasonal=True, d=None, max_d=2, D=1, max_D=2, trace=True,
                  error_action='ignore', suppress_warnings=True,
                  stepwise=True)
print(f"\nAuto ARIMA identified parameters: {model.order}, {model.seasonal_order}")

# %%
# Fit the SARIMA model on the differenced training data
model = SARIMAX(differenced_train, order=model.order, seasonal_order=model.seasonal_order, freq="MS")
model_fit = model.fit(disp=0)  # Suppress convergence output

# %%
# Make predictions on the differenced test set
predictions = model_fit.predict(start=differenced_test.index[0], end=differenced_test.index[-1])

# %%
# Invert differencing (if applied earlier)
if n_diffs > 0:
    # Invert seasonal differencing
    predictions_diff = pd.Series(predictions, index=differenced_test.index)
    predictions_seasonal = pd.Series(differenced_train.iloc[0]['Price'], index=differenced_train.index)
    predictions_seasonal = predictions_seasonal.add(predictions_diff, fill_value=0).fillna(0)

    # Invert first differencing
    predictions = pd.Series(scaled_test.iloc[0]['Price'], index=scaled_test.index)
    predictions = predictions.add(predictions_seasonal.cumsum(), fill_value=0).fillna(0)
else:
    # No differencing applied, so predictions on scaled test data is sufficient
    predictions = pd.Series(predictions, index=scaled_test.index)

# %%
# Invert scaling
predictions = scaler.inverse_transform(predictions.values.reshape(-1, 1))

# %%
# Evaluate model performance (using MSE here, consider adding other metrics)
# Ensure that 'predictions' is only as long as 'test'
if len(predictions) > len(test['Price']):
    predictions = predictions[:len(test['Price'])]

mse = mean_squared_error(test['Price'], predictions)
print(f'MSE: {mse}')
# %%
# Plot actual vs predicted values
plt.figure(figsize=(12, 5))
plt.plot(train, label='Training')
plt.plot(test, label='Test')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()

end_time = time.time()
total_time = end_time - start_time
print(f'Total time taken: {total_time} seconds')
# Additional steps (optional):
# - Save the model for future use
# - Implement error handling for potential issues

# %%
