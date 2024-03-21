# %%
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
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
# Create training and testing datasets
train, test = train_test_split(data, test_size=0.2, shuffle=False)
print(f'Training set:{train.shape} \nTesting set:{test.shape}')

# %%
train.head()

# %%
test.head()

# %%
train_start_date = '2002-01-01'
test_start_date = '2018-10-01'

# train[2002-01]

# %%
# Visualize the training and testing datasets
plt.figure(figsize=(12, 6))
plt.plot(train, label='Training')
plt.plot(test, label='Testing')
plt.title('Commodity Prices Monthly')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend()
plt.show()

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
timesteps = 5

# %%
scaled_train_data_timesteps=np.array([[j for j in scaled_train_data[i:i+timesteps]] for i in range(0,len(scaled_train_data)-timesteps+1)])[:,:,0]
scaled_train_data_timesteps.shape

# %%
scaled_test_data_timesteps=np.array([[j for j in scaled_test_data[i:i+timesteps]] for i in range(0,len(scaled_test_data)-timesteps+1)])[:,:,0]
scaled_test_data_timesteps.shape

# %%
x_train, y_train = scaled_train_data_timesteps[:,:timesteps-1],scaled_train_data_timesteps[:,[timesteps-1]]
x_test, y_test = scaled_test_data_timesteps[:,:timesteps-1],scaled_test_data_timesteps[:,[timesteps-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# %%
# check 
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)

# %%
model.fit(x_train, y_train[:,0])

# %%
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

# %%
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)

# %%
# Scaling the predictions
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

print(len(y_train_pred), len(y_test_pred))

# %%
# Scaling the original values
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

print(len(y_train), len(y_test))

# %%
train_timestamps = data[(data.index < test_start_date) & (data.index >= train_start_date)].index[timesteps-1:]
test_timestamps = data[test_start_date:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))

# %%
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()

# %%
from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(y_train_pred,y_train)
mape

# %%
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()

# %%
## check performance on full dataset

# %%
# Extracting load values as numpy array
full_data = data.copy().values

# Scaling
full_data = scaler.transform(full_data)

# Transforming to 2D tensor as per model input requirement
data_timesteps=np.array([[j for j in full_data[i:i+timesteps]] for i in range(0,len(full_data)-timesteps+1)])[:,:,0]
print("Tensor shape: ", data_timesteps.shape)

# Selecting inputs and outputs from data
X, Y = data_timesteps[:,:timesteps-1],data_timesteps[:,[timesteps-1]]
print("X shape: ", X.shape,"\nY shape: ", Y.shape)

# %%
Y_pred = model.predict(X).reshape(-1,1)

# Inverse scale and reshape
Y_pred = scaler.inverse_transform(Y_pred)
Y = scaler.inverse_transform(Y)

# %%
plt.figure(figsize=(30,8))
plt.plot(Y, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(Y_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()

# %%
from sklearn.metrics import mean_absolute_error 
print(f'MAE : {mean_absolute_error(Y, Y_pred)}') 

# %%
print(f'MAPE : {mean_absolute_percentage_error(Y_pred,Y)}%')

# %%



