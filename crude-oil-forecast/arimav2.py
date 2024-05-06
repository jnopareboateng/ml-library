#%%
# Import necessary libraries
import logging
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import pmdarima as pm 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pmdarima.arima.utils import nsdiffs,ndiffs
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Set display options
pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")
#%%

# Configure logging to print to standard output
logging.basicConfig(level=logging.INFO, format='%(message)s')
#%%

class ARIMAModel:
    def __init__(self, file_path, forecast_horizon):
        self.file_path = file_path
        self.forecast_horizon = forecast_horizon

    def load_data(self):
        try:
            data = pd.read_csv(self.file_path, parse_dates=True, index_col=[0])
            return data
        except FileNotFoundError:
            logging.error(f"File {self.file_path} not found.")
            return None
        except pd.errors.EmptyDataError:
            logging.error(f"File {self.file_path} is empty.")
            return None
        except Exception as e:
            logging.error(f"An error occurred while loading the file: {str(e)}")
            return None

    def plot_data(self, data, show_plots=True):
        data_plot = px.line(data, x=data.index, y=data['Price'], title="Brent Crude Oil Prices from 2002 -2022")
        if show_plots:
            data_plot.show()
        return data_plot

# Similar modifications can be made to the other plotting methods

    def test_stationarity(self, series):
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

    def preprocess_data(self, data):
        n_diffs = pm.arima.ndiffs(data['Price'], test='adf')
        logging.info(f"\nNumber of differences required : {n_diffs}")
        if n_diffs > 0:
            differenced_data = data.diff(n_diffs).dropna()
        else:
            differenced_data = data.copy()
        return differenced_data

    def plot_differenced_data(self, differenced_data):
        differenced_data.plot()
        plt.savefig('differenced_data.png')


    def check_seasonal_differencing(self, data):
        nsdiff= nsdiffs(data['Price'], m=12, test='ch')
        logging.info(f"Seasonal differences required: {nsdiff}")

    def plot_seasonal_decomposition(self, differenced_data):
        decomposition = seasonal_decompose(differenced_data["Price"], model="additive")
        decomposition.plot()
        plt.savefig('seasonal_decomposition.png')

    def plot_acf_pacf_plots(self, differenced_data):
        plot_acf(differenced_data['Price'], title='ACF Plot')
        plt.savefig('acf_plot.png')
        plot_pacf(differenced_data['Price'], title='PACF Plot')
        plt.savefig('pacf_plot.png')

    def evaluate_stationarity(self, differenced_data):
        logging.info("Testing stationarity of scaled training data:")
        self.test_stationarity(differenced_data['Price'])

    def auto_arima_model(self, differenced_data):
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(differenced_data):
            train, test = differenced_data.iloc[train_index], differenced_data.iloc[test_index]
        model = pm.auto_arima(train, trace=True, cv=tscv)
        order = model.order
        seasonal_order = model.seasonal_order
        logging.info(f"\nAuto ARIMA identified parameters: {order}, {seasonal_order}")
        return model, train, test, order, seasonal_order

    def interpret_model(self, model):
        logging.info(f"ARIMA Order: {model.order}")
        logging.info(f"Seasonal Order: {model.seasonal_order}")
        logging.info(f"AIC: {model.aic()}")
        logging.info(f"BIC: {model.bic()}")
        logging.info(f"HQIC: {model.hqic()}")

    def plot_residuals(self, model):
        model.plot_diagnostics(figsize=(12, 8))
        plt.savefig('residuals_plot.png')
        plt.show()

    def fit_sarimax_model(self, differenced_data, order, seasonal_order):
        model = SARIMAX(endog=differenced_data, order=order, seasonal_order=seasonal_order, freq="MS")
        results = model.fit(disp=0)
        logging.info(results.summary())
        return results

    def forecast_future_values(self, train, order, seasonal_order, n_periods):
        if 'Price' in train.columns:
            model = SARIMAX(train['Price'], order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
            forecast, stderr, conf_int = model_fit.forecast(steps=n_periods)
            return forecast
        else:
            logging.error("Train data does not have a 'Price' column.")
            return None

    def plot_forecast_with_confidence_intervals(self, history, predictions, forecast_summary_90, forecast_summary_95):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history.index, y=history, mode='lines', name='Historical Data'))
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions, mode='lines', name='Forecasted Values'))
        fig.add_trace(go.Scatter(x=forecast_summary_90.index, y=forecast_summary_90['mean_ci_upper'], mode='lines', name='90% Confidence Interval', line=dict(width=0)))
        fig.add_trace(go.Scatter(x=forecast_summary_90.index, y=forecast_summary_90['mean_ci_lower'], mode='lines', name='90% Confidence Interval', line=dict(width=0), fill='tonexty'))
        fig.add_trace(go.Scatter(x=forecast_summary_95.index, y=forecast_summary_95['mean_ci_upper'], mode='lines', name='95% Confidence Interval', line=dict(width=0)))
        fig.add_trace(go.Scatter(x=forecast_summary_95.index, y=forecast_summary_95['mean_ci_lower'], mode='lines', name='95% Confidence Interval', line=dict(width=0), fill='tonexty'))
        fig.update_layout(title='Forecast with Confidence Intervals')
        fig.show()
    def calculate_error_metrics(self, test, predictions):
        if 'Price' in test.columns:
            mae = mean_absolute_error(test['Price'], predictions)
            mape = mean_absolute_percentage_error(test['Price'], predictions)
            logging.info("""Evaluating with MAE and MAPE """)
            logging.info(f"Mean Absolute Error: {mae}")
            logging.info(f"Mean Absolute Percentage Error: {mape}")
        else:
            logging.error("Test data does not have a 'Price' column.")


