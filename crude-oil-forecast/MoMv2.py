# %%
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import uniform, reciprocal
# from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.stats import uniform, randint
import plotly.graph_objects as go
import plotly.express as px
%matplotlib inline

# %%

# Import necessary functions from codiumate_enhanced_arima.py
# from codiumate_enhanced_arima import forecast_future_values, plot_forecast_with_confidence_intervals, calculate_error_metrics

# %%
def load_data():
    file_path = 'Modified_Data.csv'
    # url = 'https://raw.githubusercontent.com/jnopareboateng/ml-library/master/crude-oil-forecast/Modified_Data.csv'
    # response = requests.get(url)
    # response.raise_for_status()
    data = pd.read_csv(file_path, parse_dates=True, index_col=[0])
    return data
# %%
def plot_data(data):
    data_plot = px.line(data, x=data.index, y=data['Price'], title="Brent Crude Oil Prices from 2002 -2022")
    # decomposition = seasonal_decompose(data["Price"], model="additive")
    # decomposition.plot()
    # plt.savefig('data_visualization.png')
    return data_plot

#%%
timesteps = 24

# SVR model

svr = SVR()

# %%
# Hyperparameter tuning for SVR
param_distributions = {
    'C': uniform(0.1, 10),
    'epsilon': uniform(0.1, 1),
    'gamma': uniform(0.1, 1)
}


# %%

# Forecast with SVR
predictions_svr, forecast_summary_90_svr, forecast_summary_95_svr = forecast_future_values(scaled_test['Price'], (0, 0, 0), (0, 0, 0, 0), horizon=24)

# Plot forecast with confidence intervals for SVR
plot_forecast_with_confidence_intervals(scaled_test['Price'], predictions_svr, forecast_summary_90_svr, forecast_summary_95_svr)

# Calculate error metrics for SVR
calculate_error_metrics(y_test, y_pred_svr)

# %%
# Random Forest Regressor
rf = RandomForestRegressor()

# %%
# ... (existing code for hyperparameter tuning and model fitting)

# Forecast with Random Forest
predictions_rf, forecast_summary_90_rf, forecast_summary_95_rf = forecast_future_values(scaled_test['Price'], (0, 0, 0), (0, 0, 0, 0), horizon=24)

# Plot forecast with confidence intervals for Random Forest
plot_forecast_with_confidence_intervals(scaled_test['Price'], predictions_rf, forecast_summary_90_rf, forecast_summary_95_rf)

# Calculate error metrics for Random Forest
calculate_error_metrics(y_test, y_pred_rf)

# %%
# XGBoost Regressor
xgb = XGBRegressor()

# %%
# ... (existing code for hyperparameter tuning and model fitting)

# Forecast with XGBoost
predictions_xgb, forecast_summary_90_xgb, forecast_summary_95_xgb = forecast_future_values(scaled_test['Price'], (0, 0, 0), (0, 0, 0, 0), horizon=24)

# Plot forecast with confidence intervals for XGBoost
plot_forecast_with_confidence_intervals(scaled_test['Price'], predictions_xgb, forecast_summary_90_xgb, forecast_summary_95_xgb)

# Calculate error metrics for XGBoost
calculate_error_metrics(y_test, y_pred_xgb)