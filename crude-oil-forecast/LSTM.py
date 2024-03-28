# %%
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.metrics import mean_squared_error
from common.preprocessor import load_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
import plotly.offline as py
import plotly.graph_objs as go

import keras_tuner as kt

import tensorflow as tf

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
train.head(10)

# %%
test.head(10)

# %%
# Prepare data for training
scaler = MinMaxScaler()
scaled_train = train.copy()
scaled_test = test.copy()
scaled_train['Price'] = scaler.fit_transform(scaled_train[['Price']])
scaled_test['Price'] = scaler.transform(scaled_test[['Price']])
print(f'Scaled Training Set: {scaled_train.shape}\nScaled Testing Set {scaled_test.shape}')

# %%
# create a function to prepare the data
def create_dataset(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# %%
# prepare the data
time_steps = 5
X_train, y_train = create_dataset(scaled_train[['Price']], scaled_train.Price,time_steps)
X_test, y_test = create_dataset(scaled_test[['Price']], scaled_test.Price,time_steps)
print(X_train.shape, y_train.shape)

# %%
LSTM??

# %%
from keras.regularizers import L1L2
from tensorflow.keras.losses import mean_squared_error
import keras

def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(mean_squared_error(y_true, y_pred))

def model_builder(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('input_unit',min_value=32,max_value=512,step=32), return_sequences=True, input_shape= ( X_train.shape[1], X_train.shape[2]), bias_regularizer = L1L2(0.009, 0.004)))
    for i in range(hp.Int('n_layers', 1, 4)):
        model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=512,step=32),return_sequences=True))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.05)))
    model.add(Dense(1, activation=hp.Choice('dense_activation',values=['relu', 'sigmoid'],default='relu')))
   
    model.compile(loss=root_mean_squared_error, optimizer='adam',metrics = ['mse'])
    
    return model

tuner = kt.RandomSearch(model_builder, objective='val_loss', max_trials = 10, executions_per_trial =1,directory = "./silver/")

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', # Monitor the validation loss
                                                     min_delta=0,    # until it doesn't change (or gets worse)
                                                     patience=5,  # patience > 1 so it continutes if it is not consistently improving
                                                     verbose=0, 
                                                     mode='auto')]

tuner.search(x=X_train, y=y_train, epochs = 200, batch_size =512, validation_data=(X_test, y_test), callbacks=[callbacks],shuffle = True)


# %%
tuner.results_summary()



# %%
# LSTM IS BAD AT EXTRAPOLATING (Data outside of training range)
best_model = tuner.get_best_models(num_models=1)[0]

# %%
history = best_model.fit(X_train, y_train, epochs=150, batch_size=128, validation_data=(X_test, y_test), shuffle=False , verbose=0) # shuffle=False because we want to keep the order of the data


# explain the parameters and the values you used


# %%
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# %%
predictions = best_model.predict(X_test)

# %%
predictions = scaler.inverse_transform(predictions.reshape(-1,1))

# %%
actual = scaler.inverse_transform(y_test.reshape(-1,1))

# %%
plt.plot(predictions, label='predict')
plt.plot(actual, label='true')
plt.legend()
plt.show()

# %%
# evaluate the model
scores = best_model.evaluate(X_train, y_train, verbose=0)
print("%s: %.2f%%" % (best_model.metrics_names[1], scores[1]*100))

# %%



