#%%
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
                                       n_iter=10, cv=5, random_state=42)
random_search_svr = RandomizedSearchCV(svr, param_distributions=param_dist_svr,
                                       n_iter=10, cv=5, random_state=42)
random_search_rf = RandomizedSearchCV(rf, param_distributions=param_dist_rf,
                                      n_iter=10, cv=5, random_state=42)
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
best_xgb.fit(X_train, y_train)
best_svr.fit(X_train, y_train)
best_rf.fit(X_train, y_train)
#%%

# Make predictions
xgb_predictions = best_xgb.predict(X_train[-24:]) # Predict the last 24 months
svr_predictions = best_svr.predict(X_train[-24:])
rf_predictions = best_rf.predict(X_train[-24:])
#%%

# Calculate the error metrics
mae_xgb = mean_absolute_error(y_train[-24:], xgb_predictions) # Calculate the MAE for the last 24 months
mae_svr = mean_absolute_error(y_train[-24:], svr_predictions)
mae_rf = mean_absolute_error(y_train[-24:], rf_predictions)

mse_xgb = mean_squared_error(y_train[-24:], xgb_predictions)
mse_svr = mean_squared_error(y_train[-24:], svr_predictions)
mse_rf = mean_squared_error(y_train[-24:], rf_predictions)

mape_xgb = mean_absolute_percentage_error(y_train[-24:], xgb_predictions)
mape_svr = mean_absolute_percentage_error(y_train[-24:], svr_predictions)
mape_rf = mean_absolute_percentage_error(y_train[-24:], rf_predictions)

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

# print(f"XGBoost MAE: {mae_xgb}")
# print(f"SVR MAE: {mae_svr}")
# print(f"Random Forest MAE: {mae_rf}")
#%%

# Plot the historical data and the forecasted data for each model individually
index = data.index

# Convert X and y to numpy arrays of type float32
# X = np.array(data.index.values.reshape(-1, 1), dtype=np.float32)
# y = np.array(data['Price'].values, dtype=np.float32)

fig_xgb = go.Figure()
fig_xgb.add_trace(go.Scatter(x=train.index[timesteps-1:], y=y_train.flatten(), mode='lines', name='Historical Data'))
fig_xgb.add_trace(go.Scatter(x=train.index[-24:], y=xgb_predictions, mode='lines', name='Forecasted Values'))
fig_xgb.update_layout(title='XGBoost: Historical Data vs Forecasted Values')
fig_xgb.show()
#%%

fig_svr = go.Figure()
fig_svr.add_trace(go.Scatter(x=train.index[timesteps-1:], y=y_train.flatten(), mode='lines', name='Historical Data'))
fig_svr.add_trace(go.Scatter(x=X_train[-24:], y=svr_predictions, mode='lines', name='Forecasted Values'))
fig_svr.update_layout(title='SVR: Historical Data vs Forecasted Values')
fig_svr.show()
#%%

fig_rf = go.Figure()
fig_rf.add_trace(go.Scatter(x=train.index[timesteps-1:], y=y_train.flatten(), mode='lines', name='Historical Data'))
fig_rf.add_trace(go.Scatter(x=X_train[-24:], y=rf_predictions, mode='lines', name='Forecasted Values'))
fig_rf.update_layout(title='Random Forest: Historical Data vs Forecasted Values')
fig_rf.show()
#%%

# Plot the historical data and the forecasted data for all models combined
fig_combined = go.Figure()
fig_combined.add_trace(go.Scatter(x=train.index[timesteps-1:], y=y_train.flatten(), mode='lines', name='Historical Data'))
fig_combined.add_trace(go.Scatter(x=train.index[timesteps-1:], y=xgb_predictions, mode='lines', name='XGBoost Forecasted Values'))
fig_combined.add_trace(go.Scatter(x=train.index[timesteps-1:], y=svr_predictions, mode='lines', name='SVR Forecasted Values'))
fig_combined.add_trace(go.Scatter(x=train.index[timesteps-1:], y=rf_predictions, mode='lines', name='Random Forest Forecasted Values'))
fig_combined.update_layout(title='All Models: Historical Data vs Forecasted Values')
fig_combined.show()
