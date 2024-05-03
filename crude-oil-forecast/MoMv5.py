#%%
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, reciprocal
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.stats import uniform, randint
%matplotlib inline

# Load the data
url = 'https://raw.githubusercontent.com/jnopareboateng/ml-library/master/crude-oil-forecast/Modified_Data.csv'
data = pd.read_csv(url, parse_dates=True, index_col=[0])

# Plot the data
data.plot(y='Price', title='Commodity Prices Monthly', figsize=(12, 6))
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.show()

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

# Initialize the SVR model
svr = SVR()

# Set the hyperparameters for the SVR model
param_distributions = {
    'C': uniform(1, 10),  # Regularization parameter
    'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5],  # Margin of tolerance
    'gamma': reciprocal(0.001, 0.1),  # Kernel coefficient
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']  # Kernel type
}

# Perform randomized search
rnd_search_cv = RandomizedSearchCV(svr, param_distributions, n_iter=10, verbose=2, cv=3)
rnd_search_cv.fit(X_train, y_train)

# Print the best parameters
print(f"Best parameters: {rnd_search_cv.best_params_}")

# Fit and predict with the best parameters
best_svr = rnd_search_cv.best_estimator_
best_svr.fit(X_train, y_train)
y_pred = best_svr.predict(X_train[-24:])

# Plot the actual vs predicted data
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index[timesteps-1:], y=y_train.flatten(), mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=train.index[-24:], y=y_pred, mode='lines', name='Predicted'))
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price')
fig.show()
