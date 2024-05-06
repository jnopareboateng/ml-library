#%% 
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import logging
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import pmdarima as pm 
import plotly.express as px
import plotly.graph_objects as go
# import requests
from plotly.subplots import make_subplots
from pmdarima.arima.utils import nsdiffs,ndiffs
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
# import mlflow

pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")
# %matplotlib inline

#%% 

# Configure logging to print to standard output
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Now, logging.info will print to the notebook
logging.info("Modeling and Forecasting Brent Crude Oil Using ARIMA ")
class ARIMAModel:
    def __init__(self, file_path, forecast_horizon):
        self.file_path = file_path
        self.forecast_horizon = forecast_horizon
    def load_data():
        file_path = 'Modified_Data.csv'
        try:
            data = pd.read_csv(file_path, parse_dates=True, index_col=[0])
            return data
        except FileNotFoundError:
            logging.error(f"File {file_path} not found.")
            return None
        except pd.errors.EmptyDataError:
            logging.error(f"File {file_path} is empty.")
            return None
        except Exception as e:
            logging.error(f"An error occurred while loading the file: {str(e)}")
            return None
    #%% 

    def plot_data(data):
        data_plot = px.line(data, x=data.index, y=data['Price'], title="Brent Crude Oil Prices from 2002 -2022")
        decomposition = seasonal_decompose(data["Price"], model="additive")
        decomposition.plot()
        plt.savefig('data_visualization.png')
        return data_plot
    #%% 

    def test_stationarity(series):
        def adf_test(series):
            result = adfuller(series, autolag='AIC')
            logging.info(f'ADF Statistic: {result[0]}')
            logging.info(f'p-value: {result[1]}')
            if result[1] <= 0.05:
                logging.info("Data is likely stationary.")
            else:
                logging.info("Data may be non-stationary. Consider differencing.")

        logging.info("""Testing stationarity of data:""")
        adf_test(series)
    #%% 

    def preprocess_data(data):
        n_diffs = pm.arima.ndiffs(data['Price'], test='adf')
        logging.info(f"\nNumber of differences required : {n_diffs}")
        if n_diffs > 0:
            differenced_data = data.diff(n_diffs).dropna()
        else:
            differenced_data = data.copy()
        return differenced_data
    #%% 

    def plot_differenced_data(differenced_data):
        differenced_data.plot()
        plt.savefig('differenced_data.png')
    #%% 

    def check_seasonal_differencing(data):
        nsdiff= nsdiffs(data['Price'], m=12, test='ch')
        logging.info(f"Seasonal differences required: {nsdiff}")
    #%% 

    def plot_seasonal_decomposition(differenced_data):
        decomposition = seasonal_decompose(differenced_data["Price"], model="additive")
        decomposition.plot()
        plt.savefig('seasonal_decomposition.png')
    #%% 

    def plot_acf_pacf_plots(differenced_data):
        plot_acf(differenced_data['Price'], title='ACF Plot')
        plt.savefig('acf_plot.png')
        plot_pacf(differenced_data['Price'], title='PACF Plot')
        plt.savefig('pacf_plot.png')
    #%% 

    def evaluate_stationarity(differenced_data):
        adf_test(differenced_data['Price'])
    #%% 
    def auto_arima_model(differenced_data):
        model = pm.auto_arima(differenced_data['Price'], trace=True)
        logging.info(f"\nAuto ARIMA identified parameters: {model.order}, {model.seasonal_order}")
        logging.info(f'model order: {model.order}, \nmodel seasonal order: {model.seasonal_order}')
        return model
    #%% 
def interpret_model(model):
    logging.info(f"ARIMA Order: {model.order}")
    logging.info(f"Seasonal Order: {model.seasonal_order}")
    logging.info(f"AIC: {model.aic()}")
    logging.info(f"BIC: {model.bic()}")
    logging.info(f"HQIC: {model.hqic()}")
    #%%
    def plot_residuals(model):
        model.plot_diagnostics(figsize=(12, 8))
        plt.savefig('residuals_plot.png')
    #%% 
    def fit_sarimax_model(differenced_data, order, seasonal_order):
        model = SARIMAX(endog=differenced_data, order=order, seasonal_order=seasonal_order, freq="MS")
        results = model.fit(disp=0)
        logging.info(results.summary())
        return results
    #%% 
    def forecast_future_values(history, order, seasonal_order, horizon):
        model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
        history = data['Price']
        horizon = 24
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=horizon)
        forecast_period = pd.date_range(start=history.index[-1], periods=horizon+1, freq='MS')[1:]
        predictions = pd.Series(predictions, index=forecast_period)
        forecast_90 = model_fit.get_forecast(steps=horizon)
        forecast_summary_90 = forecast_90.summary_frame(alpha=0.10)
        forecast_95 = model_fit.get_forecast(steps=horizon)
        forecast_summary_95 = forecast_95.summary_frame(alpha=0.05)
        plt.plot(history.index, history, label='Historical Data')
        plt.plot(predictions.index, predictions, color='red', label='Forecasted Values')
        plt.fill_between(forecast_summary_90.index,
                        forecast_summary_90['mean_ci_lower'],
                        forecast_summary_90['mean_ci_upper'], color='pink', alpha=0.1, label='90% Confidence Interval')
        plt.fill_between(forecast_summary_95.index,
                        forecast_summary_95['mean_ci_lower'],
                        forecast_summary_95['mean_ci_upper'], color='blue', alpha=0.05, label='95% Confidence Interval')
        plt.title('Forecast with Confidence Intervals')
        plt.legend()
        plt.savefig('forecast_with_confidence_intervals.png')
        return predictions, forecast_summary_90, forecast_summary_95

    def plot_forecast_with_confidence_intervals(history, predictions, forecast_summary_90, forecast_summary_95):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history.index, y=history, mode='lines', name='Historical Data'))
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions, mode='lines', name='Forecasted Values'))
        fig.add_trace(go.Scatter(x=forecast_summary_90.index, y=forecast_summary_90['mean_ci_upper'], mode='lines', name='90% Confidence Interval', line=dict(width=0)))
        fig.add_trace(go.Scatter(x=forecast_summary_90.index, y=forecast_summary_90['mean_ci_lower'], mode='lines', name='90% Confidence Interval', line=dict(width=0), fill='tonexty'))
        fig.add_trace(go.Scatter(x=forecast_summary_95.index, y=forecast_summary_95['mean_ci_upper'], mode='lines', name='95% Confidence Interval', line=dict(width=0)))
        fig.add_trace(go.Scatter(x=forecast_summary_95.index, y=forecast_summary_95['mean_ci_lower'], mode='lines', name='95% Confidence Interval', line=dict(width=0), fill='tonexty'))
        fig.update_layout(title='Forecast with Confidence Intervals')
        fig.show()
    #%% 

    def calculate_error_metrics(data, predictions):
        data= data[-len(predictions):]
        mae = mean_absolute_error(data, predictions)
        mape = mean_absolute_percentage_error(data, predictions)
        logging.info("""Evaluating with MAE and MAPE """)
        logging.info(f"Mean Absolute Error: {mae}")
        logging.info(f"Mean Absolute Percentage Error: {mape}")
#%% 
# todo: improve readability of logs
# Instantiate the ARIMAModel class
arima_model = ARIMAModel('Modified_Data.csv', 24)

# Load and validate the data
data = arima_model.load_data()
if data is None or 'Price' not in data.columns:
    logging.error("Invalid data.")
    return

# Preprocess the data
differenced_data = preprocess_data(data)
if differenced_data is None:
    logging.error("Error in data preprocessing.")
    return

# Fit the model
model = auto_arima_model(differenced_data)
if model is None:
    logging.error("Error in model fitting.")
    return

# Forecast future values
predictions, forecast_summary_90, forecast_summary_95 = forecast_future_values(data['Price'], model.order, model.seasonal_order, 24)
if predictions is None:
    logging.error("Error in forecasting.")
    return

# Calculate error metrics
mae, mape = calculate_error_metrics(data['Price'], predictions)
if mae is None or mape is None:
    logging.error("Error in error metric calculation.")
    return

# Print the error metrics
logging.info(f"Mean Absolute Error: {mae}")
logging.info(f"Mean Absolute Percentage Error: {mape}")
# %%
