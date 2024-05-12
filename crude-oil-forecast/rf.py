#%%
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
# from xgboost import XGBRegressor
# from arima import ARIMAModel
# from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
#%%
data = pd.read_csv('Modified_Data.csv', parse_dates=True, index_col=[0])
#%%
train_start_date = '2002-01-01'
test_start_date = '2019-01-01'
#%%
#%%
# set the train and test data and print the dimensions of it
train = data.copy()[(data.index >= train_start_date) & (data.index < test_start_date)][['Price']]
test = data.copy()[data.index >= test_start_date][['Price']]

print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)

# Set the timesteps
timesteps = 24

train_data = train.values
test_data = test.values

# Create timesteps for the train data
train_data_timesteps = np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]

# Split the data into features and target
X_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]

# Create timesteps for the test data
test_data_timesteps = np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
X_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

#%%
param_dist_rf = {"n_estimators": [100, 200, 300, 400, 500],
                 "max_depth": [3, None],
                 "max_features": [1, 3, 10],
                 "min_samples_split": [2, 3, 10],
                 "bootstrap": [True, False]}

rf = RandomForestRegressor(random_state=42)

random_search_rf = RandomizedSearchCV(rf, param_distributions=param_dist_rf,
                                      n_iter=10, cv=TimeSeriesSplit(n_splits=5), random_state=42)

#%%
random_search_rf.fit(X_train, y_train)
best_rf = random_search_rf.best_estimator_
#%%
best_rf.fit(train_data_timesteps[:,:timesteps-1], train_data_timesteps[:,[timesteps-1]])

#%%
rf_predictions = best_rf.predict(test_data_timesteps[:,:timesteps-1])
#%%
# Calculate the evaluation metrics
rmse_rf = np.sqrt(mean_squared_error(test_data_timesteps[:,[timesteps-1]], rf_predictions))
mae_rf = mean_absolute_error(test_data_timesteps[:,[timesteps-1]], rf_predictions)
mape_rf = mean_absolute_percentage_error(test_data_timesteps[:,[timesteps-1]], rf_predictions)

print('Random Forest RMSE:', rmse_rf)
print('Random Forest MAE:', mae_rf)
print('Random Forest MAPE:', mape_rf, '%')
#%%
# Plot the actual vs predicted prices
fig = go.Figure()
fig.add_trace(go.Scatter(x=test.index, y=test_data_timesteps[:,[timesteps-1]].flatten(), mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=test.index, y=rf_predictions, mode='lines', name='Predicted'))
fig.update_layout(title='Actual vs Predicted Prices - Random Forest',
                  xaxis_title='Date',
                  yaxis_title='Price')
fig.show()
#%%
future_dates = pd.date_range(start=data.index[-1], periods=25, freq='M')[1:] # Start from the last date in the data and generate the next 24 months

# Make predictions on the future data
rf_forecast = best_rf.predict(test_data_timesteps[-24:,:timesteps-1])

# Plot the historical data and the forecasted data for each model individually
fig_rf = go.Figure()
fig_rf.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines', name='Historical Data'))
fig_rf.add_trace(go.Scatter(x=future_dates, y=rf_forecast, mode='lines', name='Forecasted Values'))
fig_rf.update_layout(title='Random Forest: Historical Data vs Forecasted Values')
fig_rf.show()
#%%


# %%
