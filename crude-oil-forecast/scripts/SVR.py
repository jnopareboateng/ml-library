# %%
import pandas as pd
import numpy as np
from sklearn.svm import SVR
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, reciprocal
from common.preprocessor import load_data
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
%matplotlib inline
# %%
data = load_data('data', 'Commodity Prices Monthly.csv')
data.head()
# %%
data.plot(y='Price', title='Commodity Prices Monthly', figsize=(12, 6))
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.show()
# %%
# set the train and test data with start dates
train_start_date = '2002-01-01'
test_start_date = '2019-01-01'
# %%
# visualize the train and test data
data[(data.index < test_start_date) & (data.index >= train_start_date)][['Price']].rename(columns={'Price':'train'}) \
    .join(data[test_start_date:][['Price']].rename(columns={'Price':'test'}), how='outer') \
    .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.show()
# %%
# set the train and test data and print the dimensions of it
train = data.copy()[(data.index >= train_start_date) & (data.index < test_start_date)][['Price']]
test = data.copy()[data.index >= test_start_date][['Price']]

print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)
# %%
# Prepare data for training
scaler = MinMaxScaler()
scaled_train = train.copy()
scaled_test = test.copy()
scaled_train['Price'] = scaler.fit_transform(scaled_train[['Price']])
print(f'Scaled Training Set: {scaled_train.shape}\nScaled Testing Set {scaled_test.shape}')

# %%
# Convert to numpy arrays
scaled_train_data = scaled_train.values
scaled_test_data = scaled_test.values

# %%
timesteps = 3
scaled_train_data_timesteps=np.array([[j for j in scaled_train_data[i:i+timesteps]] for i in range(0,len(scaled_train_data)-timesteps+1)])[:,:,0]
scaled_train_data_timesteps.shape

# %%
scaled_test_data_timesteps=np.array([[j for j in scaled_test_data[i:i+timesteps]] for i in range(0,len(scaled_test_data)-timesteps+1)])[:,:,0]
scaled_test_data_timesteps.shape

# %%
X_train, y_train = scaled_train_data_timesteps[:,:timesteps-1],scaled_train_data_timesteps[:,[timesteps-1]]
X_test, y_test = scaled_test_data_timesteps[:,:timesteps-1],scaled_test_data_timesteps[:,[timesteps-1]]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# %%
SVR??

# %%
svr = SVR()
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
y_pred = best_svr.predict(X_test)

# Calculate the MAE and MAPE
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAE: {mae}\nMAPE: {mape}")
# %%
# plot with plotly
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=test.index[timesteps-1:], y=y_test.flatten(), mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=test.index[timesteps-1:], y=y_pred, mode='lines', name='Predicted'))
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price')
fig.show()

