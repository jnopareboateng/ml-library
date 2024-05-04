#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
#%%

def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=True, index_col=[0])
#%%

def prepare_data(data, timesteps):
    train_data = data.values
    train_data_timesteps = np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
    X_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
    return X_train, y_train
#%%

def initialize_models():
    return XGBRegressor(random_state=42), SVR(), RandomForestRegressor(random_state=42)
#%%

def initialize_search(models, param_distributions, n_iter=10, cv=5, random_state=42):
    return [RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=random_state) for model, param_dist in zip(models, param_distributions)]
#%%

def fit_search(search_objects, X_train, y_train):
    for search in search_objects:
        search.fit(X_train, y_train)
    return search_objects
#%%

def get_best_estimators(search_objects):
    return [search.best_estimator_ for search in search_objects]
#%%

def fit_estimators(estimators, X_train, y_train):
    for estimator in estimators:
        estimator.fit(X_train, y_train)
    return estimators
#%%

def calculate_error_metrics(estimators, X_train, X_test, y_train, y_test):
    error_metrics = []
    for estimator in estimators:
        estimator.fit(X_train, y_train)
        forecast = estimator.predict(X_test)
        mae = mean_absolute_error(y_test, forecast)
        mse = mean_squared_error(y_test, forecast)
        error_metrics.append((mae, mse))
    return error_metrics
#%%

def plot_data(data, forecast, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines', name='Historical Data'))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Forecasted Values'))
    fig.update_layout(title=title)
    fig.show()
#%%

def main():
    # Load and prepare data
    data = load_data('Modified_Data.csv')
    X_train, y_train = prepare_data(data, 24)

# Initialize models and search objects
    models = initialize_models()
    param_dist_xgb, param_dist_svr, param_dist_rf = initialize_param_distributions()
    param_distributions = [param_dist_xgb, param_dist_svr, param_dist_rf]
    search_objects = initialize_search(models, param_distributions)

    # Fit search objects and get best estimators
    search_objects = fit_search(search_objects, X_train, y_train)
    estimators = get_best_estimators(search_objects)

    # Fit estimators
    estimators = fit_estimators(estimators, X_train, y_train)

    # Calculate error metrics
    error_metrics = calculate_error_metrics(estimators, X_train, X_test, y_train, y_test)

    # Plot data
    for estimator, (mae, mse) in zip(estimators, error_metrics):
        forecast = estimator.predict(X_test)
        plot_data(data, forecast, f'{estimator.__class__.__name__}: Historical Data vs Forecasted Values')
#%%

if __name__ == '__main__':
    main()
# %%
