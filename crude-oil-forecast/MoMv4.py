#%%
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from xgboost import XGBRegressor
# merged.py

from arima import test_stationarity, preprocess_data, auto_arima_model, fit_sarimax_model, forecast_future_values, calculate_error_metrics as calculate_error_metrics_arima
# from MoMv4 import load_data as load_data_MoMv4, initialize_models, fit_models, prepare_data, split_data, calculate_error_metrics as calculate_error_metrics_MoMv4, generate_forecast, plot_forecasts, plot_combined_forecasts
from arima import test_stationarity,preprocess_data,plot_differenced_data,check_seasonal_differencing,plot
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
#%%
def load_data():
    file_path = 'Modified_Data.csv'
    data = pd.read_csv(file_path, parse_dates=True, index_col=[0])
    return data
# %%
def plot_data(data):
    data_plot = px.line(data, x=data.index, y=data['Price'], title="Brent Crude Oil Prices from 2002 -2022")
    return data_plot
#%%
# Define the parameter distributions for RandomizedSearchCV
param_dist_xgb = {"n_estimators": [100, 200, 300, 400, 500],
                  "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                  "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
                  "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                  "subsample": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}

param_dist_svr = {"C": [0.1, 1, 10, 100, 1000],
                  "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
                  "kernel": ['rbf']}

param_dist_rf = {"n_estimators": [100, 200, 300, 400, 500],
                 "max_depth": [3, None],
                 "max_features": [1, 3, 10],
                 "min_samples_split": [2, 3, 10],
                 "bootstrap": [True, False]}
#%%
# Prepare the data

data = load_data()
# Set the train data and print the dimensions of it
train = data.copy()[['Price']]
print('Training data shape: ', train.shape)

# Convert to numpy arrays
train_data = train.values

# Set the timesteps
timesteps = 24

# Create timesteps for the train data
train_data_timesteps = np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]

# Split the data into features and target
X_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]

#%%

# Initialize the models
xgb = XGBRegressor(random_state=42)
svr = SVR()
rf = RandomForestRegressor(random_state=42)
#%%

# Initialize the RandomizedSearchCV objects
random_search_xgb = RandomizedSearchCV(xgb, param_distributions=param_dist_xgb,
                                       n_iter=10, cv=TimeSeriesSplit(n_splits=5), random_state=42)
random_search_svr = RandomizedSearchCV(svr, param_distributions=param_dist_svr,
                                       n_iter=10, cv=TimeSeriesSplit(n_splits=5), random_state=42)
random_search_rf = RandomizedSearchCV(rf, param_distributions=param_dist_rf,
                                      n_iter=10, cv=TimeSeriesSplit(n_splits=5), random_state=42)
#%%

# Fit the RandomizedSearchCV objects to the data
random_search_xgb.fit(X_train, y_train)
random_search_svr.fit(X_train, y_train)
random_search_rf.fit(X_train, y_train)
#%%

# Get the best estimators
best_xgb = random_search_xgb.best_estimator_
best_svr = random_search_svr.best_estimator_
best_rf = random_search_rf.best_estimator_
#%%

# Fit the best estimators to the data
best_xgb.fit(train_data_timesteps[:,:timesteps-1], train_data_timesteps[:,[timesteps-1]])
best_svr.fit(train_data_timesteps[:,:timesteps-1], train_data_timesteps[:,[timesteps-1]])
best_rf.fit(train_data_timesteps[:,:timesteps-1], train_data_timesteps[:,[timesteps-1]])
#%%

# Calculate the error metrics
# Split the data into training and test sets
train_size = int(len(train_data_timesteps) * 0.8)
X_train, X_test = train_data_timesteps[:train_size,:timesteps-1], train_data_timesteps[train_size:,:timesteps-1]
y_train, y_test = train_data_timesteps[:train_size,[timesteps-1]], train_data_timesteps[train_size:,[timesteps-1]]

# Fit the best estimators to the training data
best_xgb.fit(X_train, y_train)
best_svr.fit(X_train, y_train)
best_rf.fit(X_train, y_train)

# Make predictions on the test set
xgb_forecast = best_xgb.predict(X_test)
svr_forecast = best_svr.predict(X_test)
rf_forecast = best_rf.predict(X_test)

# Calculate the error metrics
mae_xgb = mean_absolute_error(y_test, xgb_forecast)
mae_svr = mean_absolute_error(y_test, svr_forecast)
mae_rf = mean_absolute_error(y_test, rf_forecast)

mse_xgb = mean_squared_error(y_test, xgb_forecast)
mse_svr = mean_squared_error(y_test, svr_forecast)
mse_rf = mean_squared_error(y_test, rf_forecast)

mape_xgb = mean_absolute_percentage_error(y_test, xgb_forecast)
mape_svr = mean_absolute_percentage_error(y_test, svr_forecast)
mape_rf = mean_absolute_percentage_error(y_test, rf_forecast)

print(f"XGBoost MAE: {mae_xgb}")
print(f"SVR MAE: {mae_svr}")
print(f"Random Forest MAE: {mae_rf}")

print(f"XGBoost MSE: {mse_xgb}")
print(f"SVR MSE: {mse_svr}")
print(f"Random Forest MSE: {mse_rf}")

print(f"XGBoost MAPE: {mape_xgb}")
print(f"SVR MAPE: {mape_svr}")
print(f"Random Forest MAPE: {mape_rf}")
#%%

# Generate future timestamps
future_dates = pd.date_range(start=data.index[-1], periods=25, freq='M')[1:] # Start from the last date in the data and generate the next 24 months

# Make predictions
# Forecast the future values
xgb_forecast = best_xgb.predict(train_data_timesteps[-24:,:timesteps-1]) # Use the last 24 months to forecast the next 24 months
svr_forecast = best_svr.predict(train_data_timesteps[-24:,:timesteps-1])
rf_forecast = best_rf.predict(train_data_timesteps[-24:,:timesteps-1])

#%%

# Plot the historical data and the forecasted data for each model individually
index = data.index

fig_xgb = go.Figure()
fig_xgb.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines', name='Historical Data'))
fig_xgb.add_trace(go.Scatter(x=future_dates, y=xgb_forecast, mode='lines', name='Forecasted Values'))
fig_xgb.update_layout(title='XGBoost: Historical Data vs Forecasted Values')
fig_xgb.show()
#%%

fig_svr = go.Figure()
fig_svr.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines', name='Historical Data'))
fig_svr.add_trace(go.Scatter(x=future_dates, y=svr_forecast, mode='lines', name='Forecasted Values'))
fig_svr.update_layout(title='SVR: Historical Data vs Forecasted Values')
fig_svr.show()
#%%

fig_rf = go.Figure()
fig_rf.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines', name='Historical Data'))
fig_rf.add_trace(go.Scatter(x=future_dates, y=rf_forecast, mode='lines', name='Forecasted Values'))
fig_rf.update_layout(title='Random Forest: Historical Data vs Forecasted Values')
fig_rf.show()

#%%

# Plot the historical data and the forecasted data for all models combined
fig_combined = go.Figure()
fig_combined.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines', name='Historical Data'))
fig_combined.add_trace(go.Scatter(x=future_dates, y=xgb_forecast, mode='lines', name='XGBoost Forecasted Values'))
fig_combined.add_trace(go.Scatter(x=future_dates, y=svr_forecast, mode='lines', name='SVR Forecasted Values'))
fig_combined.add_trace(go.Scatter(x=future_dates, y=rf_forecast, mode='lines', name='Random Forest Forecasted Values'))
fig_combined.update_layout(title='All Models: Historical Data vs Forecasted Values')
fig_combined.show()

# %%
