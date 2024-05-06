import mlflow
import mlflow.sklearn
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima 
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from arima import load_data,test_stationarity, preprocess_data, plot_differenced_data, check_seasonal_differencing,plot_seasonal_decomposition,plot_acf_pacf_plots

# Set up MLflow tracking
mlflow.set_tracking_uri("path/to/tracking/uri")
mlflow.set_experiment("Time Series Forecasting With Mixture of Models")

def prepare_data(data, train_start_date, test_start_date, tscv=None):
    """
    Prepare the data for training and testing.
    """
    # TimeSeriesSplit
    data = load_data()
    tscv = TimeSeriesSplit(n_splits=5)
    if tscv:
        for train_index, test_index in tscv.split(data):
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]
            break
    else:
        train_data = data.loc[:train_start_date]
        test_data = data.loc[test_start_date:]

    return train_data, test_data

def visualize_data(train_data, test_data):
    """
    Visualize the train-test split.
    """
    data = load_data()
    data_plot = px.line(data, x=data.index, y=data['Price'], title="Brent Crude Oil Prices from 2002 -2022")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_data.index, y=train_data['Price'], mode='lines', name='Training Data'))
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Price'], mode='lines', name='Test Data'))
    fig.update_layout(title='Train-Test Split Visualization')
    fig.show()
    return data_plot, fig

def initialize_models(train_data,random_state=42):
    """
    Initialize the forecasting models.
    """
    arima_model = auto_arima(train_data['Price'], trace=True)
    xgb_model = XGBRegressor(random_state=random_state)
    svr_model = SVR()
    rf_model = RandomForestRegressor(random_state=random_state)

    mlflow.log_param("arima_order", arima_model.order)
    mlflow.log_param("arima_seasonal_order", arima_model.seasonal_order)

    return arima_model, xgb_model, svr_model, rf_model

def arima_processor():
    """
    Preprocess the data for ARIMA.
    """
    return test_stationarity, preprocess_data, plot_differenced_data, check_seasonal_differencing,plot_seasonal_decomposition,plot_acf_pacf_plots

def train_models(train_data, differenced_data, timesteps):
    """
    Train the forecasting models.
    """
    arima_model.fit(differenced_data)

    X_train, y_train = prepare_data(train_data, timesteps=24)
    xgb_model.fit(X_train, y_train)
    svr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

def forecast_future(train_data, test_data, arima_model, xgb_model, svr_model, rf_model, timesteps):
    """
    Make forecasts for future time steps.
    """
    arima_forecast = forecast_future_values(train_data['Price'], arima_model.order, arima_model.seasonal_order, len(test_data))

    X_test = prepare_data(test_data, timesteps=24)[0]
    xgb_forecast = xgb_model.predict(X_test)
    svr_forecast = svr_model.predict(X_test)
    rf_forecast = rf_model.predict(X_test)

    return arima_forecast, xgb_forecast, svr_forecast, rf_forecast

def visualize_individual_models(train_data, test_data, arima_forecast, xgb_forecast, svr_forecast, rf_forecast):
    """
    Visualize the forecasts from each model.
    """
    figs = plot_forecasts(train_data, test_data.index, {
        'ARIMA': arima_forecast,
        'XGBoost': xgb_forecast,
        'SVR': svr_forecast,
        'Random Forest': rf_forecast
    })
    return figs

def calculate_error_metrics(test_data, arima_forecast, xgb_forecast, svr_forecast, rf_forecast):
    """
    Evaluate the performance of each model using appropriate metrics.
    """
    error_metrics = []
    error_metrics.append(calculate_error_metrics(test_data['Price'], arima_forecast, 'ARIMA'))
    error_metrics.append(calculate_error_metrics(test_data['Price'], xgb_forecast, 'XGBoost'))
    error_metrics.append(calculate_error_metrics(test_data['Price'], svr_forecast, 'SVR'))
    error_metrics.append(calculate_error_metrics(test_data['Price'], rf_forecast, 'Random Forest'))

    error_metrics_df = pd.concat(error_metrics, ignore_index=True)

    mlflow.log_metric("arima_mae", error_metrics_df.loc[error_metrics_df['Model'] == 'ARIMA', 'MAE'].values[0])
    mlflow.log_metric("xgboost_mae", error_metrics_df.loc[error_metrics_df['Model'] == 'XGBoost', 'MAE'].values[0])
    mlflow.log_metric("svr_mae", error_metrics_df.loc[error_metrics_df['Model'] == 'SVR', 'MAE'].values[0])
    mlflow.log_metric("rf_mae", error_metrics_df.loc[error_metrics_df['Model'] == 'Random Forest', 'MAE'].values[0])

    return error_metrics_df

def visualize_combined_models(train_data, test_data, arima_forecast, xgb_forecast, svr_forecast, rf_forecast):
    """
    Create a visualization comparing forecasts from all models.
    """
    combined_fig = plot_combined_forecasts(train_data, test_data.index, {
        'ARIMA': arima_forecast,
        'XGBoost': xgb_forecast,
        'SVR': svr_forecast,
        'Random Forest': rf_forecast
    })
    return combined_fig

def main(data, train_start_date, test_start_date, tscv=None, timesteps=24):
    mlflow.start_run()

    train_data, test_data = prepare_data(data, train_start_date, test_start_date, tscv)
    visualize_data(train_data, test_data)

    arima_model, xgb_model, svr_model, rf_model = initialize_models()
    differenced_data = preprocess_data(train_data)
    train_models(train_data, differenced_data, timesteps)

    arima_forecast, xgb_forecast, svr_forecast, rf_forecast = forecast_future(train_data, test_data, arima_model, xgb_model, svr_model, rf_model, timesteps)

    individual_figs = visualize_individual_models(train_data, test_data, arima_forecast, xgb_forecast, svr_forecast, rf_forecast)
    error_metrics_df = calculate_error_metrics(test_data, arima_forecast, xgb_forecast, svr_forecast, rf_forecast)
    combined_fig = visualize_combined_models(train_data, test_data, arima_forecast, xgb_forecast, svr_forecast, rf_forecast)

    mlflow.log_metric("arima_mape", error_metrics_df.loc[error_metrics_df['Model'] == 'ARIMA', 'MAPE'].values[0])
    mlflow.log_metric("xgboost_mape", error_metrics_df.loc[error_metrics_df['Model'] == 'XGBoost', 'MAPE'].values[0])
    mlflow.log_metric("svr_mape", error_metrics_df.loc[error_metrics_df['Model'] == 'SVR', 'MAPE'].values[0])
    mlflow.log_metric("rf_mape", error_metrics_df.loc[error_metrics_df['Model'] == 'RF', 'RF'].values[0])