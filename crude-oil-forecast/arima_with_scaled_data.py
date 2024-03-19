import pandas as pd
import numpy as np
import warnings
import pmdarima as pm
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
data = pd.read_csv('Modified Data.csv', parse_dates=True, index_col=[0])

# %%
data.head()

# %%
data.plot(y='Price', title='Commodity Prices Monthly', figsize=(12, 6))
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.show()

# %%
# create training and testing datasets
train, test = train_test_split(data, test_size=0.2, shuffle=False)

# %%
train.shape, test.shape

# %%
# visualize the training and testing datasets
plt.figure(figsize=(12, 6))
plt.plot(train, label='Training')
plt.plot(test, label='Testing')
plt.title('Commodity Prices Monthly')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend()
plt.show()

# %%
# make a copy of the original training and testing datasets
scaled_train = train.copy()
scaled_test = test.copy()

# prepare data for training
scaler = MinMaxScaler()
scaled_train['Price'] = scaler.fit_transform(scaled_train[['Price']])
scaled_train.head()

# %%
# Plot original data
plt.figure(figsize=(12, 6))
plt.hist(data['Price'], bins=30, alpha=0.5, label='Original')
plt.title('Histogram of Original Commodity Prices')
plt.xlabel('Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.show()

# %%
# Plot scaled data
plt.figure(figsize=(12, 6))
plt.hist(scaled_train['Price'], bins=30, alpha=0.5, label='Scaled')
plt.title('Histogram of Scaled Commodity Prices')
plt.xlabel('Scaled Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.show()

# %%
# scale test data
scaled_test['Price'] = scaler.transform(scaled_test[['Price']])
scaled_test.head()

# %%
# check to see if data is stationary

def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'No. of lags used: {result[2]}')
    print(f'No. of observations used: {result[3]}')
    print('Critical Values:')
    for k, v in result[4].items():
        print(f'   {k}: {v}')
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")

adf_test(scaled_train['Price'])

# %%
adf_test(scaled_test['Price'])

# %%
# Assuming `data` is your time series data
n_diffs = pm.arima.ndiffs(scaled_train['Price'], test='adf')  # 'adf' for Augmented Dickey-Fuller test

print(f"Number of differences required: {n_diffs}")

# %%
# Assuming `data` is your time series data
n_diffs = pm.arima.ndiffs(scaled_test['Price'], test='adf')  # 'adf' for Augmented Dickey-Fuller test

print(f"Number of differences required: {n_diffs}")

# %%
# acf and pacf plots for ar(1) and ma(1) processes

# AR(1) process
plot_acf(scaled_train['Price'], lags=20, title='ACF for AR(1) process')
plt.show()
plot_pacf(scaled_train['Price'], lags=20, title='PACF for AR(1) process')
plt.show()

# %%
# use auto arima to find the best parameters for training
model = auto_arima(scaled_train['Price'], start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, max_d=2, D=1, max_D=2, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)

# %%
# First difference
train_diff = scaled_train.diff().dropna()
train_diff.head()

# %%
# Seasonal difference
# Assuming the seasonality is 12 (e.g., for monthly data), adjust as necessary
train_seasonal_diff = train_diff.diff(12).dropna()
train_seasonal_diff.head()

# %%
# Do the same for the test set
test_diff = scaled_test.diff().dropna()
test_diff.head()

# %%
test_seasonal_diff = test_diff.diff(12).dropna()
test_seasonal_diff.head()

# %%
# use auto arima to find the best parameters
model = auto_arima(train_seasonal_diff, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, max_d=2, D=1, max_D=2, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
model.fit(train_seasonal_diff)

# %%

model = SARIMAX(train_seasonal_diff, order=(3, 1, 1), seasonal_order=(2, 1, 0, 12), freq="MS")
model_fit = model.fit(disp=0)  # disp=0 to suppress convergence output to avoid clutter in the notebook

# %%
# Make predictions on the differenced seasonal test dataset.
predictions = model_fit.predict(start=test_seasonal_diff.index[0], end=test_seasonal_diff.index[-1])

# Reverse the seasonal differencing.
predictions_diff = pd.Series(predictions, index=test_seasonal_diff.index)
predictions_seasonal = pd.Series(test_diff.iloc[0], index=test_diff.index)
predictions_seasonal = predictions_seasonal.add(predictions_diff, fill_value=0).fillna(0)

# Reverse the first differencing.
predictions = pd.Series(test.iloc[0], index=test.index)
predictions = predictions.add(predictions_seasonal.cumsum(), fill_value=0).fillna(0)

# Reverse the scaling.
predictions = scaler.inverse_transform(predictions.values.reshape(-1, 1))

# Convert the predictions to a Series.
predictions = pd.Series(predictions.flatten(), index=test.index)

# %%
test.head()

# %%
import matplotlib.pyplot as plt

# Calculate the MSE
mse = mean_squared_error(test, predictions)
print(f'MSE: {mse}')

# %%
# Plot the results
plt.figure(figsize=(12, 5))
plt.plot(train, label='Training')
plt.plot(test, label='Test')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
