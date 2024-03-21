# %%
# IMPORT NEEDED LIBRARIES 
import pandas as pd
import numpy as np
import warnings
import pmdarima as pm
import math
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from common.preprocessor import load_data
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
%matplotlib inline
pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")

# %%
# load data from the preprocessor and set index to date column
data = pd.read_csv('Modified Data.csv', parse_dates=True, index_col=[0])

# %%
data.head() # display the first 5 rows of the data

# %%
data.describe() # display the summary statistics of the data

# %%
# visualize the data
data.plot(y='Price', title='Commodity Prices Monthly', figsize=(12, 6))
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.show()

# %%
# visualize the components of the data
decomposition = seasonal_decompose(data["Price"], model="additive")  # "Price" is likely your column name for oil prices
decomposition.plot()  # Visualize the trend, seasonal component, and residuals

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
# Check for stationarity with ADF test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("Data is likely stationary.")
    else:
        print("Data may be non-stationary. Consider differencing.")

print("Testing stationarity of scaled training data:")
adf_test(train['Price'])

# %%
# Identify number of differences required (if necessary)
n_diffs = pm.arima.ndiffs(train['Price'], test='adf')
print(f"\nNumber of differences required for scaled training data: {n_diffs}")


# %%
# Perform differencing if required
if n_diffs > 0:
    differenced_train = train.diff(n_diffs).dropna()
else:
    differenced_train = train.copy()

# %%
# plot differenced data
differenced_train.plot()

# %%
# visualize the components of the differenced data
decomposition = seasonal_decompose(differenced_train["Price"], model="additive")  # "Price" is likely your column name for oil prices
decomposition.plot()  # Visualize the trend, seasonal component, and residuals


# %%
# ACF and PACF plots (optional) using lags of 60 (5 years)
plot_acf(differenced_train['Price'], lags=60, title='ACF Plot')
plt.show()
plot_pacf(differenced_train['Price'], lags=60, title='PACF Plot ')
plt.show()

# %%


df_2002 = data['2002']
df_2003 = data['2003']
df_2004 = data['2004']
df_2005 = data['2005']
df_2006 = data['2006']
# Create subplot figure
fig = make_subplots(rows=5, cols=1)

# Add traces
fig.add_trace(go.Scatter(x=df_2002.index, y=df_2002['Price'], name='Price in 2002'), row=1, col=1)
fig.add_trace(go.Scatter(x=df_2003.index, y=df_2003['Price'], name='Price in 2003'), row=2, col=1)
fig.add_trace(go.Scatter(x=df_2004.index, y=df_2004['Price'], name='Price in 2004'), row=3, col=1)
fig.add_trace(go.Scatter(x=df_2005.index, y=df_2005['Price'], name='Price in 2005'), row=4, col=1)
fig.add_trace(go.Scatter(x=df_2006.index, y=df_2006['Price'], name='Price in 2006'), row=5, col=1)

# Update xaxis properties
fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_xaxes(title_text="Date", row=3, col=1)
fig.update_xaxes(title_text="Date", row=4, col=1)
fig.update_xaxes(title_text="Date", row=5, col=1)

# Update yaxis properties
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Price", row=2, col=1)
fig.update_yaxes(title_text="Price", row=3, col=1)
fig.update_yaxes(title_text="Price", row=4, col=1)
fig.update_yaxes(title_text="Price", row=5, col=1)

# Update layout
fig.update_layout(height=1000, width=1200, title_text="Price from 2002 to 2006")

fig.show()

# %%
# Check for stationarity on differenced data with ADF test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("Data is likely stationary.")
    else:
        print("Data may be non-stationary. Consider differencing.")

print("Testing stationarity of scaled training data:")
adf_test(differenced_train['Price'])

# %%
# ACF and PACF plots on differenced data
plot_acf(differenced_train['Price'], lags=60, title='ACF Plot')
plt.show()
plot_pacf(differenced_train['Price'], lags=60, title='PACF Plot')
plt.show()

# %%
# Use auto_arima to find best parameters
model = auto_arima(differenced_train['Price'], start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                  start_P=0, seasonal=True, d=None, max_d=2, D=1, max_D=2, trace=True,
                  error_action='ignore', suppress_warnings=True,
                  stepwise=True)
print(f"\nAuto ARIMA identified parameters: {model.order}, {model.seasonal_order}")
# decide tradeoff between time and aic

# %%
# other information criteria
model = auto_arima(differenced_train['Price'], start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                  start_P=0, seasonal=True, d=None, max_d=2, D=1, max_D=2, trace=True,
                  error_action='ignore', suppress_warnings=True,
                  stepwise=True, information_criterion='bic')
print(f"\nAuto ARIMA identified parameters: {model.order}, {model.seasonal_order}")
# decide tradeoff between time and aic

# %%
# other information criteria
model = auto_arima(differenced_train['Price'], start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                  start_P=0, seasonal=True, d=None, max_d=2, D=1, max_D=2, trace=True,
                  error_action='ignore', suppress_warnings=True,
                  stepwise=True, information_criterion='hqic')
print(f"\nAuto ARIMA identified parameters: {model.order}, {model.seasonal_order}")
# decide tradeoff between time and aic

# %%
print(f'model order: {model.order}, \nmodel seasonal order: {model.seasonal_order}')

# %%
# Fit the SARIMA model on the differenced training data
model = SARIMAX(endog=differenced_train, order=model.order, seasonal_order=model.seasonal_order, freq="MS")
results = model.fit(disp=0)  # Suppress convergence output
print(results.summary())

# %%
test.shape

# %%
HORIZON = 3
test_shifted = test.copy()

for t in range(1, HORIZON):
    test_shifted['Price+'+str(t)] = test_shifted['Price'].shift(-t, freq='MS')

test_shifted = test_shifted.dropna(how='any')
test_shifted.head(5)

# %%
model.order

# %%
model.seasonal_order

# %%
%%time
training_window = 40
# dedicate 24 months (2 years) for training

train_ts = train['Price']
test_ts = test_shifted

history = [x for x in train_ts]
history = history[(-training_window):]

predictions = list()

order = model.order
seasonal_order = model.seasonal_order

for t in range(test_ts.shape[0]):
    model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    yhat = model_fit.forecast(steps = HORIZON)
    predictions.append(yhat)
    obs = list(test_ts.iloc[t])
    # move the training window
    history.append(obs[0])
    history.pop(0)
    print(test_ts.index[t])
    print(t+1, ': predicted =', yhat, 'expected =', obs)


# %%
eval_df = pd.DataFrame(predictions, columns=['m+'+str(t) for t in range(1, HORIZON+1)])
eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='month')
eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
# eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
eval_df.head()

# %%
# print one step forecast MAPE
print(f'''
    One Step forecast MAPE: 
    {mean_absolute_percentage_error(eval_df[eval_df['month'] == 'm+1']['prediction'], eval_df[eval_df['month'] == 'm+1']['actual'])}''')


# %%
# print multistep mape
print(f'''
    Multi-Step forecast MAPE: 
    {mean_absolute_percentage_error(eval_df['prediction'], eval_df['actual'])}''')

# %%
# plot of actual vs predicted
if(HORIZON == 1):
    ## Plotting single step forecast
    eval_df.plot(x='timestamp', y=['actual', 'prediction'], style=['r', 'b'], figsize=(15, 8))

else:
    ## Plotting multi step forecast
    plot = eval_df[(eval_df.month=='m+1')][['timestamp', 'actual']]
    for m in range(1, HORIZON+1):
        plot['m+'+str(m)] = eval_df[(eval_df.month=='m+'+str(m))]['prediction'].values

    fig = plt.figure(figsize=(15, 8))
    plt.plot(plot['timestamp'], plot['actual'], color='red', linewidth=4.0, label='Actual')  # Add label here
    for m in range(1, HORIZON+1):
        x = plot['timestamp'][(m-1):]
        y = plot['m+'+str(m)][0:len(x)]
        plt.plot(x, y, color='blue', linewidth=4*math.pow(.9,m), alpha=math.pow(0.8,m), label='Prediction m+'+str(m))  # Add label here
    
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend()  # Add this line to display the legend
plt.show()

# %%
import plotly.graph_objects as go

# Create a new figure
fig = go.Figure()

if(HORIZON == 1):
    ## Plotting single step forecast
    fig.add_trace(go.Scatter(x=eval_df['timestamp'], y=eval_df['actual'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=eval_df['timestamp'], y=eval_df['prediction'], mode='lines', name='Prediction'))

else:
    ## Plotting multi step forecast
    plot = eval_df[(eval_df.month=='m+1')][['timestamp', 'actual']]
    for m in range(1, HORIZON+1):
        plot['m+'+str(m)] = eval_df[(eval_df.month=='m+'+str(m))]['prediction'].values

    fig.add_trace(go.Scatter(x=plot['timestamp'], y=plot['actual'], mode='lines', name='Actual'))
    for m in range(1, HORIZON+1):
        x = plot['timestamp'][(m-1):]
        y = plot['m+'+str(m)][0:len(x)]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Prediction m+'+str(m)))

# Set the title and labels
fig.update_layout(title='Actual vs Predicted', xaxis_title='Timestamp', yaxis_title='Price', height=1000, width=1000)


# Show the figure
fig.show()


