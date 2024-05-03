#%%
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import logging
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, reciprocal
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import plotly.graph_objects as go

%matplotlib inline

#%%

file_path = 'Modified_Data.csv'
data = pd.read_csv(file_path, parse_dates=True, index_col=[0])
data.head()

#%%

data.plot(y='Price', title='Commodity Prices Monthly', figsize=(12, 6))
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.show()
#%%

train_start_date = '2002-01-01'
test_start_date = '2019-01-01'
#%%

data[(data.index < test_start_date) & (data.index >= train_start_date)][['Price']].rename(columns={'Price':'train'}) \
    .join(data[test_start_date:][['Price']].rename(columns={'Price':'test'}), how='outer') \
    .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.show()
#%%

train = data.copy()[(data.index >= train_start_date) & (data.index < test_start_date)][['Price']]
test = data.copy()[data.index >= test_start_date][['Price']]

logging.info('Training data shape: %s', train.shape)
logging.info('Test data shape: %s', test.shape)
#%%

scaler = MinMaxScaler()
scaled_train = train.copy()
scaled_train['Price'] = scaler.fit_transform(scaled_train[['Price']])
scaled_test = test.copy()
scaled_test['Price'] = scaler.transform(scaled_test[['Price']])

logging.info('Scaled Training Set: %s', scaled_train.shape)
logging.info('Scaled Testing Set %s', scaled_test.shape)
#%%

scaled_train_data = scaled_train.values
scaled_test_data = scaled_test.values
#%%

timesteps = 3

scaled_train_data_timesteps = np.array([scaled_train_data[i:i+timesteps] for i in range(0,len(scaled_train_data)-timesteps+1)])[:,:,0]
scaled_train_data_timesteps.shape
#%%

scaled_train_data
#%%

scaled_test_data_timesteps = np.array([scaled_test_data[i:i+timesteps] for i in range(0,len(scaled_test_data)-timesteps+1)])[:,:,0]
scaled_test_data_timesteps.shape
#%%

X_train, y_train = scaled_train_data_timesteps[:,:timesteps-1],scaled_train_data_timesteps[:,[timesteps-1]]
X_test, y_test = scaled_test_data_timesteps[:,:timesteps-1],scaled_test_data_timesteps[:,[timesteps-1]]

logging.info('X_train shape: %s', X_train.shape)
logging.info('y_train shape: %s', y_train.shape)
logging.info('X_test shape: %s', X_test.shape)
logging.info('y_test shape: %s', y_test.shape)
#%%

svr = SVR()
#%%

param_distributions = {
    'C': uniform(1, 10),  # Regularization parameter
    'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5],  # Margin of tolerance
    'gamma': reciprocal(0.001, 0.1),  # Kernel coefficient
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']  # Kernel type
}

cv_folds = min(5, len(X_train))
rnd_search_cv = RandomizedSearchCV(svr, param_distributions, n_iter=10, verbose=2, cv=cv_folds)
rnd_search_cv.fit(X_train, y_train)

logging.info("Best parameters: %s", rnd_search_cv.best_params_)
#%%

best_svr = rnd_search_cv.best_estimator_
y_pred = best_svr.predict(X_test)

y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_pred_inv
#%%

fig = go.Figure()
fig.add_trace(go.Scatter(x=test.index[timesteps-1:], y=y_test.flatten(), mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=test.index[timesteps-1:], y=y_pred, mode='lines', name='Predicted'))
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price')
fig.show()
#%%

rf = RandomForestRegressor()

param_distributions_rf = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(1, 10),
    'min_samples_split': uniform(0.01, 0.2),
    'min_samples_leaf': uniform(0.01, 0.2),
}

rnd_search_cv_rf = RandomizedSearchCV(rf, param_distributions_rf, n_iter=10, verbose=2, cv=3)
rnd_search_cv_rf.fit(X_train, y_train)

logging.info("Random Forest - Best parameters: %s", rnd_search_cv_rf.best_params_)
#%%

best_rf = rnd_search_cv_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
#%%

mse_rf = mean_squared_error(y_test, y_pred_rf)
logging.info("Random Forest - MSE: %s", mse_rf)
#%%

fig.add_trace(go.Scatter(x=test.index[timesteps-1:], y=y_pred_rf, mode='lines', name='Random Forest'))
#%%

xgb = XGBRegressor()

param_distributions_xgb = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(1, 10),
    'learning_rate': uniform(0.01, 0.6),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
}

rnd_search_cv_xgb = RandomizedSearchCV(xgb, param_distributions_xgb, n_iter=10, verbose=2, cv=3)
rnd_search_cv_xgb.fit(X_train, y_train)

logging.info("XGBoost - Best parameters: %s", rnd_search_cv_xgb.best_params_)
#%%

best_xgb = rnd_search_cv_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
#%%
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb)
logging.info("XGBoost - MAE: %s", mae_xgb)
logging.info("XGBoost - MAPE: %s", mape_xgb)
#%%

fig.add_trace(go.Scatter(x=test.index[timesteps-1:], y=y_pred_xgb, mode='lines', name='XGBoost'))
fig.add_trace(go.Scatter(x=test.index[timesteps-1:], y=y_pred, mode='lines', name='SVR'))
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price')


# %%
