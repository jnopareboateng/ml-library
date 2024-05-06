# merged.py

from arima import load_data as load_data_arima, plot_data, test_stationarity, preprocess_data, auto_arima_model, fit_sarimax_model, forecast_future_values, calculate_error_metrics as calculate_error_metrics_arima
from MoMv4 import load_data as load_data_MoMv4, initialize_models, fit_models, prepare_data, split_data, calculate_error_metrics as calculate_error_metrics_MoMv4, generate_forecast, plot_forecasts, plot_combined_forecasts