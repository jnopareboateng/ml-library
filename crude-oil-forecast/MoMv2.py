#%%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

#%%
def load_data(file_path):
    """
    Load data from a given file path.
    
    Parameters:
    file_path (str): The path to the data file.
    
    Returns:
    data (pd.DataFrame): The loaded data.
    """
    data = pd.read_csv(file_path, parse_dates=True, index_col=[0])
    return data

#%%
def plot_data(data):
    """
    Plot the data.
    
    Parameters:
    data (pd.DataFrame): The data to plot.
    
    Returns:
    data_plot (plotly.graph_objects.Figure): The plotly figure object.
    """
    data_plot = px.line(data, x=data.index, y=data['Price'], title="Brent Crude Oil Prices from 2002 -2022")
    return data_plot

#%%
def initialize_models(random_state=42):
    """
    Initialize the models.
    
    Parameters:
    random_state (int): The random state for model initialization.
    
    Returns:
    models (list): The list of initialized models.
    """
    models = [XGBRegressor(random_state=random_state), SVR(), RandomForestRegressor(random_state=random_state)]
    return models

#%%
def fit_models(models, X_train, y_train):
    """
    Fit the models to the data.
    
    Parameters:
    models (list): The list of models to fit.
    X_train (np.array): The training features.
    y_train (np.array): The training target.
    """
    for model in models:
        model.fit(X_train, y_train)

#%%
def calculate_error_metrics(models, X_test, y_test):
    """
    Calculate the error metrics for each model.
    
    Parameters:
    models (list): The list of models.
    X_test (np.array): The test features.
    y_test (np.array): The test target.
    """
    error_metrics = {}
    for model in models:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        error_metrics[type(model).__name__] = {'MAE': mae, 'MSE': mse, 'MAPE': mape}
    return error_metrics

#%%
def prepare_data(data, timesteps):
    train_data_timesteps = np.array([[j for j in data[i:i+timesteps]] for i in range(0, len(data)-timesteps+1)])[:,:,0]
    X_train, y_train = train_data_timesteps[:,:timesteps-1], train_data_timesteps[:,[timesteps-1]]
    return X_train, y_train

#%%
def split_data(train_data_timesteps, timesteps):
    """
    Split the data into training and test sets.
    
    Parameters:
    train_data_timesteps (np.array): The data with timesteps.
    timesteps (int): The number of timesteps.
    
    Returns:
    X_train (np.array): The training features.
    X_test (np.array): The test features.
    y_train (np.array): The training target.
    y_test (np.array): The test target.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        train_data_timesteps[:, :timesteps-1],  # Features
        train_data_timesteps[:, timesteps-1],   # Target
        test_size=0.2, random_state=42
    )
    y_train = y_train.ravel()  # Ensuring y_train is 1D
    y_test = y_test.ravel()    # Ensuring y_test is 1D
    return X_train, X_test, y_train, y_test

#%%
def generate_forecast(models, train_data_timesteps, timesteps):
    """
    Generate future forecasts.
    
    Parameters:
    models (list): The list of models.
    train_data_timesteps (np.array): The data with timesteps.
    timesteps (int): The number of timesteps.
    
    Returns:
    forecasts (dict): The forecasts for each model.
    """
    forecasts = {}
    for model in models:
        forecast = model.predict(train_data_timesteps[-timesteps:,:timesteps-1])
        forecasts[type(model).__name__] = forecast
    return forecasts

#%%
def plot_forecasts(data, future_dates, forecasts):
    """
    Plot the historical data and the forecasted data for each model individually.
    
    Parameters:
    data (pd.DataFrame): The historical data.
    future_dates (pd.DatetimeIndex): The future dates for forecasting.
    forecasts (dict): The forecasts for each model.
    """
    index = data.index
    for model_name, forecast in forecasts.items():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=index, y=data['Price'], mode='lines', name='Historical Data'))
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Forecasted Values'))
        fig.update_layout(title=f'{model_name}: Historical Data vs Forecasted Values')
        fig.show()

#%%
def plot_combined_forecasts(data, future_dates, forecasts):
    """
    Plot the historical data and the forecasted data for all models combined.
    
    Parameters:
    data (pd.DataFrame): The historical data.
    future_dates (pd.DatetimeIndex): The future dates for forecasting.
    forecasts (dict): The forecasts for each model.
    """
    index = data.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=index, y=data['Price'], mode='lines', name='Historical Data'))
    for model_name, forecast in forecasts.items():
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name=f'{model_name} Forecasted Values'))
    fig.update_layout(title='All Models: Historical Data vs Forecasted Values')
    fig.show()

#%%
def main():
    file_path = 'Modified_Data.csv'
    data = load_data(file_path)
    plot_data(data)
    train = data.copy()[['Price']]
    print('Training data shape: ', train.shape)
    train_data = train.values
    timesteps = 24
    X_train, y_train = prepare_data(train_data, timesteps)
    models = initialize_models(random_state=42)
    fit_models(models, X_train, y_train)
    train_data_timesteps = np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
    X_train, X_test, y_train, y_test = split_data(train_data_timesteps, timesteps)
    error_metrics = calculate_error_metrics(models, X_test, y_test)
    for model_name, metrics in error_metrics.items():
        print(f"{model_name} MAE: {metrics['MAE']:.3f}")
        print(f"{model_name} MSE: {metrics['MSE']:.3f}")
        print(f"{model_name} MAPE: {metrics['MAPE']:.3f}")
    future_dates = pd.date_range(start=data.index[-1], periods=25, freq='M')[1:]
    forecasts = generate_forecast(models, train_data_timesteps, timesteps)
    plot_forecasts(data, future_dates, forecasts)
    plot_combined_forecasts(data, future_dates, forecasts)

if __name__ == "__main__":
    main()
# %%
