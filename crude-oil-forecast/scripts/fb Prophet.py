# %%
import pandas as pd
import matplotlib.pyplot as plt
import itertools
# import randomized search
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from common.preprocessor import load_data
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# %%
# Load dataset
data = load_data('./data','Commodity Prices Monthly.csv')

# %%
data.head()

# %%
data = data.reset_index()

# %%
data.head()

# %%
data.columns = ['ds', 'y']

# %%
data

# %%
data.dtypes

# %%
data.describe() # display the summary statistics of the data


# %%
# plot the data
data.plot(x='ds', y= 'y', title='Commodity Prices Monthly', figsize=(12, 6))
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.show()

# %%
# set training and testing start dates
train = data[['ds', 'y']].copy()
test = data[['ds', 'y']].copy()

# %%
train = train[train['ds'] < '2019-01-01']
test = test[test['ds'] >= '2019-01-01']

train.head()

# %%
test.head()
# %%
Prophet??

# %%
# from sklearn.base import BaseEstimator
# # perform hyperparameter tuning with randomized search
# # define the parameter grid
# param_grid = {
#     'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
#     'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
#     'seasonality_mode': ['additive', 'multiplicative'],
#     'n_changepoints': [10, 20, 30, 40, 50],
#     'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
#     'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
#     'seasonality_mode': ['additive', 'multiplicative'],
#     'daily_seasonality': [True, False],
#     'weekly_seasonality': [True, False],
#     'yearly_seasonality': [True, False]
# }

# # initialize the model
# class ProphetEstimator(BaseEstimator):
#     def __init__(self, **kwargs):
#         self.model = Prophet(**kwargs)
    
#     def fit(self, X, y=None):
#         self.model.fit(X, y)
#         return self
    
#     def predict(self, X):
#         return self.model.predict(X)
# # {'seasonality_prior_scale': 0.1, 'seasonality_mode': 'multiplicative', 'n_changepoints': 10, 'holidays_prior_scale': 0.1, 'daily_seasonality': False, 'weekly_seasonality': False, 'yearly_seasonality': True, 'changepoint_prior_scale': 0.1}

# # initialize the model
# model = ProphetEstimator(seasonality_prior_scale=0.1, seasonality_mode='multiplicative', n_changepoints=10, holidays_prior_scale=0.1, daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True, changepoint_prior_scale=0.1)

# # perform grid search
# grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)
# grid_search.fit(train)


# %%

# Initialize the Prophet model
model = Prophet()

# %%
# Fit the model with your training data
model.fit(train)

# %%
# Create an empty dataframe to hold your future predictions
future = model.make_future_dataframe(periods=len(test), freq= 'MS')

# Use the model to make predictions
forecast = model.predict(future)

# Plot the original data and the forecast
model.plot(forecast)
plt.show()


# %%
model.plot_components(forecast)

# %%
forecast[['ds','yhat']]

# %%
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(data['y'], forecast[['yhat']])
print('MAE: %.3f' % mae)

# %%
mean_absolute_percentage_error(data['y'], forecast[['yhat']])


