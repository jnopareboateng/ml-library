import pandas as pd
import numpy as np
import pmdarima as pm
import warnings
from pmdarima import auto_arima
from pmdarima.arima.utils import nsdiffs
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

# Suppress warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)

def load_data(url):
    try:
        data = pd.read_csv(url, parse_dates=True, index_col=[0])
        if data.isnull().sum().any():
            data.fillna(data.mean(), inplace=True)  # Handling missing values by filling with mean
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def check_stationarity(series):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    return result[1] <= 0.05

def fit_sarimax(data, order, seasonal_order, train_size_ratio=0.8):
    train_size = int(len(data) * train_size_ratio)
    train, test = data[0:train_size], data[train_size:]
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=0)
    predictions = results.predict(start=len(train), end=len(train) + len(test) - 1)
    mse = mean_squared_error(test, predictions)
    mape = mean_absolute_percentage_error(test, predictions)
    return mse, mape, results

def plot_forecast(model_fit, history, steps=24):
    forecast = model_fit.get_forecast(steps=steps)
    forecast_index = pd.date_range(start=history.index[-1], periods=steps+1, freq='MS')[1:]
    forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)
    lower_series = pd.Series(forecast.conf_int()['lower Price'], index=forecast_index)
    upper_series = pd.Series(forecast.conf_int()['upper Price'], index=forecast_index)

    plt.figure(figsize=(10, 5))
    plt.plot(history, label='Historical Data')
    plt.plot(forecast_series, color='red', label='Forecasted Values')
    plt.fill_between(forecast_index, lower_series, upper_series, color='pink', alpha=0.1, label='Confidence Interval')
    plt.title('Forecast with Confidence Intervals')
    plt.legend()
    plt.show()

# Main execution
data_url = 'https://raw.githubusercontent.com/jnopareboateng/ml-library/master/crude-oil-forecast/Modified_Data.csv'
data = load_data(data_url)
if data is not None:
    if check_stationarity(data['Price']):
        print("Data is stationary")
    else:
        print("Data is not stationary, differencing needed")
    auto_model = auto_arima(data['Price'], trace=True, error_action='ignore', suppress_warnings=True)
    mse, mape, fitted_model = fit_sarimax(data, auto_model.order, auto_model.seasonal_order)
    print(f"MSE: {mse}, MAPE: {mape}")
    plot_forecast(fitted_model, data['Price'])