from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Define the models and their respective parameter distributions
models = {
    "XGBoost": {
        "model": XGBRegressor(objective='reg:squarederror'),
        "params": {
            "n_estimators": [100, 500, 1000],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7]
        }
    },
    "SVR": {
        "model": SVR(),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ['linear', 'rbf']
        }
    },
    "RandomForest": {
        "model": RandomForestRegressor(),
        "params": {
            "n_estimators": [100, 500, 1000],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10]
        }
    }
}

# Prepare the data
X = data.index.values.reshape(-1, 1)
y = data['Price'].values

# Use TimeSeriesSplit for time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Perform RandomizedSearchCV for each model
for name, model in models.items():
    random_search = RandomizedSearchCV(model["model"], model["params"], cv=tscv, n_iter=10, scoring='neg_mean_absolute_error')
    random_search.fit(X, y)
    print(f"Best parameters for {name}: {random_search.best_params_}")
    print(f"Best score for {name}: {-random_search.best_score_}")

#%%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import numpy as np

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

# Initialize the models
xgb = XGBRegressor(random_state=42)
svr = SVR()
rf = RandomForestRegressor(random_state=42)

# Initialize the RandomizedSearchCV objects
random_search_xgb = RandomizedSearchCV(xgb, param_distributions=param_dist_xgb,
                                       n_iter=10, cv=5, iid=False, random_state=42)
random_search_svr = RandomizedSearchCV(svr, param_distributions=param_dist_svr,
                                       n_iter=10, cv=5, iid=False, random_state=42)
random_search_rf = RandomizedSearchCV(rf, param_distributions=param_dist_rf,
                                      n_iter=10, cv=5, iid=False, random_state=42)

# Fit the RandomizedSearchCV objects to the data
random_search_xgb.fit(X, y)
random_search_svr.fit(X, y)
random_search_rf.fit(X, y)

# Get the best estimators
best_xgb = random_search_xgb.best_estimator_
best_svr = random_search_svr.best_estimator_
best_rf = random_search_rf.best_estimator_

# Fit the best estimators to the data
best_xgb.fit(X, y)
best_svr.fit(X, y)
best_rf.fit(X, y)

# Make predictions
xgb_predictions = best_xgb.predict(X)
svr_predictions = best_svr.predict(X)
rf_predictions = best_rf.predict(X)

# Calculate the error metrics
mae_xgb = mean_absolute_error(y, xgb_predictions)
mae_svr = mean_absolute_error(y, svr_predictions)
mae_rf = mean_absolute_error(y, rf_predictions)

print(f"XGBoost MAE: {mae_xgb}")
print(f"SVR MAE: {mae_svr}")
print(f"Random Forest MAE: {mae_rf}")
