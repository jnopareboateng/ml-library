# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from common.preprocessor import load_data
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

# Load data
data = load_data('data', 'Commodity Prices Monthly.csv')

# Create training and testing datasets
train, test = train_test_split(data, test_size=0.2, shuffle=False)

# Prepare data for training
scaler = MinMaxScaler()
scaled_train = train.copy()
scaled_test = test.copy()
scaled_train['Price'] = scaler.fit_transform(scaled_train[['Price']])

# Convert to numpy arrays
scaled_train_data = scaled_train.values
scaled_test_data = scaled_test.values

timesteps = 5
scaled_train_data_timesteps=np.array([[j for j in scaled_train_data[i:i+timesteps]] for i in range(0,len(scaled_train_data)-timesteps+1)])[:,:,0]
scaled_test_data_timesteps=np.array([[j for j in scaled_test_data[i:i+timesteps]] for i in range(0,len(scaled_test_data)-timesteps+1)])[:,:,0]

x_train, y_train = scaled_train_data_timesteps[:,:timesteps-1],scaled_train_data_timesteps[:,[timesteps-1]]
x_test, y_test = scaled_test_data_timesteps[:,:timesteps-1],scaled_test_data_timesteps[:,[timesteps-1]]

# Define a function to train and predict
def train_and_predict(model, x_train, y_train, x_test, scaler):
    model.fit(x_train, y_train[:,0])
    y_train_pred = model.predict(x_train).reshape(-1,1)
    y_test_pred = model.predict(x_test).reshape(-1,1)

    # Scaling the predictions
    y_train_pred = scaler.inverse_transform(y_train_pred)
    y_test_pred = scaler.inverse_transform(y_test_pred)

    return y_train_pred, y_test_pred

# SVR
svr_model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
y_train_pred_svr, y_test_pred_svr = train_and_predict(svr_model, x_train, y_train, x_test, scaler)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
y_train_pred_rf, y_test_pred_rf = train_and_predict(rf_model, x_train, y_train, x_test, scaler)

# XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=0)
y_train_pred_xgb, y_test_pred_xgb = train_and_predict(xgb_model, x_train, y_train, x_test, scaler)

# Calculate MAPE for each model
mape_svr = mean_absolute_percentage_error(y_train_pred_svr, y_train)
mape_rf = mean_absolute_percentage_error(y_train_pred_rf, y_train)
mape_xgb = mean_absolute_percentage_error(y_train_pred_xgb, y_train)

print(f'SVR MAPE: {mape_svr}%')
print(f'Random Forest Regressor MAPE: {mape_rf}%')
print(f'XGBoost Regressor MAPE: {mape_xgb}%')

# Plotting actual vs predicted for each model
plt.figure(figsize=(14, 7))

# SVR
plt.subplot(3, 1, 1)
plt.plot(y_test, color='blue', label='Actual')
plt.plot(y_test_pred_svr, color='red', label='Predicted')
plt.title('SVR')
plt.legend()

# Random Forest
plt.subplot(3, 1, 2)
plt.plot(y_test, color='blue', label='Actual')
plt.plot(y_test_pred_rf, color='red', label='Predicted')
plt.title('Random Forest')
plt.legend()

# XGBoost
plt.subplot(3, 1, 3)
plt.plot(y_test, color='blue', label='Actual')
plt.plot(y_test_pred_xgb, color='red', label='Predicted')
plt.title('XGBoost')
plt.legend()

plt.tight_layout()
plt.show()

# Forecasting for the next 5 months
def forecast_next_months(model, x_test, scaler, months=5):
    forecast = x_test[-1].tolist()
    for _ in range(months):
        new_pred = model.predict(np.array([forecast[-4:]]))
        forecast.append(new_pred[0])
    return scaler.inverse_transform([forecast[-months:]])[0]

# SVR
svr_forecast = forecast_next_months(svr_model, x_test, scaler)
print(f'SVR forecast for next 5 months: {svr_forecast}')

# Random Forest
rf_forecast = forecast_next_months(rf_model, x_test, scaler)
print(f'Random Forest forecast for next 5 months: {rf_forecast}')

# XGBoost
xgb_forecast = forecast_next_months(xgb_model, x_test, scaler)
print(f'XGBoost forecast for next 5 months: {xgb_forecast}')

from datetime import datetime, timedelta

# Get the last date in the dataset
last_date = data.index[-1]

# Generate dates for the next 5 months
next_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 6)]

# Create a DataFrame for the forecasted values
forecast_df = pd.DataFrame({
    'Date': next_dates,
    'SVR': svr_forecast,
    'Random Forest': rf_forecast,
    'XGBoost': xgb_forecast
})

# Set Date as the index
forecast_df.set_index('Date', inplace=True)

print(forecast_df)