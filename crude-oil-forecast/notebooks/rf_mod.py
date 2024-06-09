#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.graph_objects as go
from sklearn.model_selection import RandomizedSearchCV
#%%

# Load data (assuming 'Modified_Data.csv' is in the same directory)
data = pd.read_csv("../Modified_Data.csv", parse_dates=True, index_col=[0])
#%%

# Define training and testing periods
train_start_date = "2002-01-01"
test_start_date = "2019-01-01"

# Split data into training and testing sets
# set the train and test data and print the dimensions of it
train = data.copy()[(data.index >= train_start_date) & (data.index < test_start_date)][['Price']]
test = data.copy()[data.index >= test_start_date][['Price']]

print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)

#%%
# Plot train and test splits
fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index, y=train['Price'], mode='lines', name='Train'))
fig.add_trace(go.Scatter(x=test.index, y=test['Price'], mode='lines', name='Test'))
fig.update_layout(title='Train and Test Split', xaxis_title='Date', yaxis_title='Price')
fig.show()


#%%
# Convert the data to numpy array
train_data = train.values
test_data = test.values

# split the data into X and y train and test splits
X_train, y_train = train_data[:-1], train_data[1:]
X_test, y_test = test_data[:-1], test_data[1:]

y_train = y_train.ravel()
y_test = y_test.ravel()

#%%

param_dist_rf = {"n_estimators": [450, 470, 490],
                 "max_depth": [1, 3, None],
                 "max_features": ['auto', 'sqrt'],
                 "min_samples_split": [2, 5, 10],
                 "bootstrap": [True, False]}

rf = RandomForestRegressor(random_state=42) # Instantiate the model
random_search_rf = RandomizedSearchCV(rf, param_distributions=param_dist_rf, cv=5, verbose=2, random_state=42)  # Instantiate the GridSearchCV with verbose and random_state parameters


# Create and fit RandomizedSearchCV (no TimeSeriesSplit)
# rf = RandomForestRegressor(random_state=42)
# random_search_rf = RandomizedSearchCV(rf, param_distributions=param_dist_rf, n_iter=10, cv=5, random_state=42)
random_search_rf.fit(X_train, y_train)

# Get the best model from RandomizedSearchCV
best_rf = random_search_rf.best_estimator_
#%%

# Make predictions on test data
rf_predictions = best_rf.predict(X_test)

#%%

# Evaluate model performance
mse_rf = mean_squared_error(y_test, rf_predictions)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, rf_predictions)
mape_rf = mean_absolute_percentage_error(y_test, rf_predictions) * 100
da_rf = np.mean(np.sign(y_test[1:] - y_test[:-1]) == np.sign(rf_predictions[1:] - rf_predictions[:-1])) * 100

print("Random Forest Evaluation:")
print(f"  - MSE: {mse_rf:.3f}")
print(f"  - RMSE: {rmse_rf:.3f}")
print(f"  - MAE: {mae_rf:.3f}")
print(f"  - MAPE: {mape_rf:.3f}%")
print(f"  - Directional Accuracy: {da_rf:.3f}%")

#%%
# Create a list of error metric names and values
metrics = ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error', 'Mean Abs. Percentage Error', 'Directional Accuracy']
values = [mse_rf, rmse_rf, mae_rf, mape_rf, da_rf]

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.barh(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'pink', 'purple'])
for i, v in enumerate(values):
    plt.text(v, i, f'{v:.3f}', color='black', va='center', rotation=270, fontweight='bold', fontsize=10)
plt.xlabel('Error Values')
plt.ylabel('Error Metrics')
plt.title('Error Metrics Comparison')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()

# Display the chart
plt.show()


#%%

# Plot the actual vs predicted prices with plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=test.index, y=test_data.flatten(), mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=test.index, y=rf_predictions, mode='lines', name='Predicted'))
fig.update_layout(title='Actual vs Predicted Prices - Random Forest',
                  xaxis_title='Date',
                  yaxis_title='Price')
fig.show()
# %%
#%%

future_dates = pd.date_range(start=data.index[-1], periods=25, freq='M')[1:] # Start from the last date in the data and generate the next 24 months

# Make predictions on the future data
rf_forecast = best_rf.predict(test_data[-24:])

# Plot the historical data and the forecasted data for each model individually
fig_rf = go.Figure()
fig_rf.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines', name='Historical Data'))
fig_rf.add_trace(go.Scatter(x=future_dates, y=rf_forecast, mode='lines', name='Forecasted Values'))
fig_rf.update_layout(title='Random Forest: Historical Data vs Forecasted Values')
fig_rf.show()


# %%
