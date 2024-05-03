#%%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
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
timesteps = 24
X, y = [], []

# Create sequences of observations and next values to predict
for i in range(timesteps, len(data)):
    X.append(data['Price'].iloc[i-timesteps:i].values)
    y.append(data['Price'].iloc[i])

# Convert lists to numpy arrays
X, y = np.array(X), np.array(y)
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
random_search_xgb.fit(X, y)
random_search_svr.fit(X, y)
random_search_rf.fit(X, y)
#%%

# Get the best estimators
best_xgb = random_search_xgb.best_estimator_
best_svr = random_search_svr.best_estimator_
best_rf = random_search_rf.best_estimator_
#%%

# Fit the best estimators to the data
best_xgb.fit(X, y)
best_svr.fit(X, y)
best_rf.fit(X, y)
#%%

# Make predictions
xgb_predictions = best_xgb.predict(X)
svr_predictions = best_svr.predict(X)
rf_predictions = best_rf.predict(X)
#%%

# Calculate the error metrics
mae_xgb = mean_absolute_error(y, xgb_predictions)
mae_svr = mean_absolute_error(y, svr_predictions)
mae_rf = mean_absolute_error(y, rf_predictions)
#%%

print(f"XGBoost MAE: {mae_xgb}")
print(f"SVR MAE: {mae_svr}")
print(f"Random Forest MAE: {mae_rf}")
#%%

# Plot the historical data and the forecasted data for each model individually
index = data.index

# Convert X and y to numpy arrays of type float32
# X = np.array(data.index.values.reshape(-1, 1), dtype=np.float32)
# y = np.array(data['Price'].values, dtype=np.float32)

fig_xgb = go.Figure()
fig_xgb.add_trace(go.Scatter(x=X.index[timesteps-1:], y=y.flatten(), mode='lines', name='Historical Data'))
fig_xgb.add_trace(go.Scatter(x=X.index[timesteps-1:], y=xgb_predictions, mode='lines', name='Forecasted Values'))
fig_xgb.update_layout(title='XGBoost: Historical Data vs Forecasted Values')
fig_xgb.show()
#%%

fig_svr = go.Figure()
fig_svr.add_trace(go.Scatter(x=X, y=y, mode='lines', name='Historical Data'))
fig_svr.add_trace(go.Scatter(x=X, y=svr_predictions, mode='lines', name='Forecasted Values'))
fig_svr.update_layout(title='SVR: Historical Data vs Forecasted Values')
fig_svr.show()
#%%

fig_rf = go.Figure()
fig_rf.add_trace(go.Scatter(x=X.index, y=y, mode='lines', name='Historical Data'))
fig_rf.add_trace(go.Scatter(x=X.index, y=rf_predictions, mode='lines', name='Forecasted Values'))
fig_rf.update_layout(title='Random Forest: Historical Data vs Forecasted Values')
fig_rf.show()
#%%

# Plot the historical data and the forecasted data for all models combined
fig_combined = go.Figure()
fig_combined.add_trace(go.Scatter(x=X.index, y=y, mode='lines', name='Historical Data'))
fig_combined.add_trace(go.Scatter(x=X.index, y=xgb_predictions, mode='lines', name='XGBoost Forecasted Values'))
fig_combined.add_trace(go.Scatter(x=X.index, y=svr_predictions, mode='lines', name='SVR Forecasted Values'))
fig_combined.add_trace(go.Scatter(x=X.index, y=rf_predictions, mode='lines', name='Random Forest Forecasted Values'))
fig_combined.update_layout(title='All Models: Historical Data vs Forecasted Values')
fig_combined.show()
