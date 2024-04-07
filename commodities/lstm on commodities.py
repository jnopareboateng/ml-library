# %%
# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# # 1. Importing  Libraries

# %%
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization

import plotly.offline as py
import plotly.graph_objs as go

import keras_tuner as kt

import tensorflow as tf

from IPython.display import SVG
import os

import datetime, time

from tensorflow import keras

# import tensorflow as tf

import random

from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model
from keras.regularizers import L1L2

# %maptlotlib inline

# %% [markdown]
# ## Objective
# The aim of this kernal is to train an LSTM model to predict the future price of Commodities like Crude Oil, Gold ,Silver etc.  based on past time series data. This kernal will use LSTM model from the Keras Library
# 

# %% [markdown]
# ## What is LSTM?¶
# LSTM stands for long short-term memory networks, used in the field of Deep Learning. It is a variety of recurrent neural networks (RNNs) that are capable of learning long-term dependencies, especially in sequence prediction problems. LSTM has feedback connections, i.e., it is capable of processing the entire sequence of data, apart from single data points such as images. This finds application in speech recognition, machine translation, etc. LSTM is a special kind of RNN, which shows outstanding performance on a large variety of problems.

# %%
# tf.device('/device:GPU:0')


# %%
df=pd.read_csv("commodities_12_22.csv")
df.head()

# %%
df.info()

# %%
df.describe().T

# %%
df.shape

# %%
df.dtypes

# %%
df=df.iloc[::-1] # reverse the dataframe

df.head()

# %%
df[["Year", "Month", "Day"]] = df["Date"].str.split("-", expand = True)
df.head()

# %%
df_copy=df.copy()

# %%
da=["Month", "Day"]
for j,i in enumerate(da):
    col_date=df.pop(i)
    df.insert(j+2,i,col_date)
df.head()

# %%
for i in df:
    if i!="Date":
        df[i]=df[i].astype('float64')
        print(df[i])
    else:
        pass

# %%
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df

# %%
df.dtypes

# %% [markdown]
# #  2. Exploratory Data Analysis

# %%
plt.rcParams["figure.figsize"] = (15,8)
sns.lineplot(data=df, x="Date", y="Gold")
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
# ax = plt.gca()
plt.show()

# %%
plt.rcParams["figure.figsize"] = (25,8)
sns.lineplot(data=df, x="Date", y="Silver")
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
ax = plt.gca()

plt.show()

# %%
plt.rcParams["figure.figsize"] = (25,8)
sns.lineplot(data=df, x="Date", y="Crude Oil")
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
ax = plt.gca()

plt.show()

# %%
plt.rcParams["figure.figsize"] = (25,8)
sns.lineplot(data=df, x="Date", y="Brent Oil")
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
ax = plt.gca()

plt.show()

# %%
plt.rcParams["figure.figsize"] = (25,8)
sns.lineplot(data=df, x="Date", y="Natural Gas")
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
ax = plt.gca()
plt.show()

# %%
plt.rcParams["figure.figsize"] = (25,8)
sns.lineplot(data=df, x="Date", y="Copper")
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
ax = plt.gca()

plt.show()

# %%
# plt.plot??

# %%
df.corr()

# %% [markdown]
# ## Correlation Between the Commodities for the  last 10 Years

# %%

plt.figure(figsize=(12,12))
sns.heatmap(df.corr(method='pearson'),cbar=True,cmap='BuPu',annot=True)

# %% [markdown]
# * Crude Oil prices is positively correlated with Brent Oil,Natural Gas, Silver and Copper.
# * Brent Oil and Crude Oil are highly positive correlated.
# * Brent Oil is  positively correlated with Natural Gas , Silver and Copper.
# * Natural Gas is  positively correlated with Brent Oil , Crude Oil and Copper.
# * Gold prices is  positively correlated with Silver and Copper.
# * Silver is positively correlated with Brent Oil , Crude Oil , Gold and Copper.
# * Copper is positively correlated with Brent Oil , Crude Oil , Gold and Copper.
# * Copper is positively correlated with all others.
# 

# %% [markdown]
# ## Correlation Between the Commodities for the  last 5 Years

# %%
df_last5=df[df["Year"]>=2017]
df_last5.head()

# %%
df_last5[["Crude Oil","Brent Oil","Natural Gas","Gold","Silver","Copper"]].corr(method='pearson')

# %%
plt.figure(figsize=(12,12))
sns.heatmap(df_last5[["Crude Oil","Brent Oil","Natural Gas","Gold","Silver","Copper"]].corr(method='pearson'),cbar=True,cmap='BuPu',annot=True)

# %% [markdown]
# ###  From Last 5 Years
# * Crude Oil prices is highly positively correlated with Brent Oil and Natural Gas 
# * Crude Oil prices is positively correlated with Copper.
# * Brent Oil and Crude Oil are highly positive correlated.
# * Brent Oil is  positively correlated with Natural Gas and Copper.
# * Natural Gas is  positively correlated with Brent Oil , Crude Oil and Copper.
# * Gold prices is highly positively correlated with Silver 
# * Gold prices is  positively correlated with Silver and Copper.
# * Silver is positively correlated with Gold and Copper.
# * Copper is positively correlated with Brent Oil , Crude Oil , Gold and Copper.
# * Copper is positively correlated with all others.

# %%
sns.scatterplot??

# %%
plt.figure(figsize=(35,35))
j=0

for i in enumerate(df): 
    if i[1]!="Gold" and i[1]!="Date":
        j+=1
#         print(i[0])
#         plt.figure(figsize=(10,10))
        plt.subplot(7,7,j+1)
        sns.scatterplot(y=df[i[1]],x=df["Gold"])
        plt.title('Relationship between Gold and '+str(i[1]))



# %%
plt.figure(figsize=(35,35))
j=0
for i in enumerate(df): 
    if i[1]!="Silver" and i[1]!="Date":
        j+=1
        plt.subplot(7,7,j+1)
        sns.scatterplot(y=df[i[1]],x=df["Silver"])
        plt.title('Relationship between Silver and '+str(i[1]))

# %%
plt.figure(figsize=(40,40))
j=0
for i in enumerate(df): 
    if i[1]!="Crude Oil" and i[1]!="Date":
        j+=1
        plt.subplot(7,7,j+2)
        sns.scatterplot(y=df[i[1]],x=df["Crude Oil"])
        plt.title('Relationship between Crude Oil and '+str(i[1]))

# %%
plt.figure(figsize=(40,40))
j=0
for i in enumerate(df): 
    if i[1]!="Brent Oil" and i[1]!="Date":
        j+=1
        plt.subplot(7,7,j+2)
        sns.scatterplot(y=df[i[1]],x=df["Brent Oil"])
        plt.title('Relationship between Brent Oil and '+str(i[1]))

# %%
plt.figure(figsize=(40,40))
j=0
for i in enumerate(df): 
    if i[1]!="Natural Gas" and i[1]!="Date":
        j+=1
        plt.subplot(7,7,j+2)
        sns.scatterplot(y=df[i[1]],x=df["Natural Gas"])
        plt.title('Relationship between Natural Gas and '+str(i[1]))

# %%
plt.figure(figsize=(40,40))
j=0
for i in enumerate(df): 
    if i[1]!="Copper" and i[1]!="Date":
        j+=1
        plt.subplot(7,7,j+2)
        sns.scatterplot(y=df[i[1]],x=df["Copper"])
        plt.title('Relationship between Copper and '+str(i[1]))

# %% [markdown]
# # 3. Data  Cleaning,Preprocessing and Training
# We will take the last 500 days data as testing data and the remaining will be used to train the model.

# %% [markdown]
# #  Gold Prices Forecasting Using LSTM

# %%
df_Gold=df[['Date','Gold']]
df_Gold.head()

# %%
df_Gold=df_Gold.dropna()

# %%
df_Gold.info()

# %%
prediction_days = 500
df_train_g= df_Gold['Gold'][:len(df_Gold['Gold'])-prediction_days].values.reshape(-1,1)
df_test_g= df_Gold['Gold'][len(df_Gold['Gold'])-prediction_days:].values.reshape(-1,1)

scaler_train = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler_train.fit_transform(df_train_g)

scaler_test = MinMaxScaler(feature_range=(0, 1))
scaled_test = scaler_test.fit_transform(df_test_g)

# %% [markdown]
# The use of prior time steps to predict the next time step is called the sliding window method. For short, it may be called the window method in some literature. In statistics and time series analysis, this is called a lag or lag method.
# 
# The number of previous time steps is called the window width or size of the lag.
# 
# Here we have used a window of 30 days.

# %%
def create_dataset(dataset, look_back=30):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(scaled_train)
testX, testY = create_dataset(scaled_test)

# %%
trainX.shape


# %%
testX.shape


# %%
testX[150]

# %%
trainX[0]

# %%
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# %%
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
def model_builder(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('input_unit',min_value=32,max_value=512,step=32), return_sequences=True, input_shape= ( trainX.shape[1], trainX.shape[2]), bias_regularizer = L1L2(0.009, 0.004)))
    for i in range(hp.Int('n_layers', 1, 4)):
        model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=512,step=32),return_sequences=True))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.05)))
    model.add(Dense(1, activation=hp.Choice('dense_activation',values=['relu', 'sigmoid'],default='relu')))
   
    model.compile(loss=root_mean_squared_error, optimizer='adam',metrics = ['mse'])
    
    return model

tuner = kt.RandomSearch(model_builder, objective='val_loss', max_trials = 10, executions_per_trial =1,directory = "./gold2/")

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', # Monitor the validation loss
                                                     min_delta=0,    # until it doesn't change (or gets worse)
                                                     patience=5,  # patience > 1 so it continutes if it is not consistently improving
                                                     verbose=0, 
                                                     mode='auto')]

tuner.search(x=trainX, y=trainY, epochs = 200, batch_size =512, validation_data=(testX, testY), callbacks=[callbacks],shuffle = True)


# %%
tuner.results_summary()

# %%
best_model = tuner.get_best_models(num_models=1)[0]

# %%
history = best_model.fit(x=trainX, y=trainY, epochs = 150, batch_size =128, validation_data=(testX, testY), shuffle=False, verbose=0)

# %%
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# %%
predicted_gold_price = best_model.predict(testX)

# %%
predicted_gold_price = scaler_test.inverse_transform(predicted_gold_price.reshape(-1, 1))

# %%
true = scaler_test.inverse_transform(testY.reshape(-1, 1))

# %%
plt.plot(predicted_gold_price, label='predict')
plt.plot(true, label='true')
plt.legend()
plt.show()

# %%
# evaluate the model
scores = best_model.evaluate(trainX, trainY, verbose=0)
print("%s: %.2f%%" % (best_model.metrics_names[1], scores[1]*100))

# %%
best_model.save("models/goldPriceModel.h5")
print("Saved model to disk")

# %%
# load json and create model
# load model
model_g = load_model("models/goldPriceModel.h5", custom_objects={'root_mean_squared_error':                   
root_mean_squared_error})
print("dd")
# summarize model.
model_g.summary()
score = model_g.evaluate(trainX, trainY, verbose=0)
print("%s: %.2f%%" % (model_g.metrics_names[1], score[1]*100))

# %% [markdown]
# # Silver Prices Forecasting Using LSTM

# %%
df_Silver=df[['Date','Silver']]
df_Silver.head()

# %%
df_Silver=df_Silver.dropna()

# %%
df_Silver.info()

# %%
prediction_days = 500
df_train_S= df_Silver['Silver'][:len(df_Silver['Silver'])-prediction_days].values.reshape(-1,1)
df_test_S= df_Silver['Silver'][len(df_Silver['Silver'])-prediction_days:].values.reshape(-1,1)

scaler_train = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler_train.fit_transform(df_train_S)

scaler_test = MinMaxScaler(feature_range=(0, 1))
scaled_test = scaler_test.fit_transform(df_test_S)

# %% [markdown]
#  The use of prior time steps to predict the next time step is called the sliding window method. For short, it may be called the window method in some literature. In statistics and time series analysis, this is called a lag or lag method.
# 
# The number of previous time steps is called the window width or size of the lag.
# 
# Here we have used a window of 30 days.

# %%
trainX=[]
trainY=[]

# %%
def create_dataset(dataset, look_back=30):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(scaled_train)
testX, testY = create_dataset(scaled_test)

# %%
trainX.shape


# %%
trainY.shape

# %%
testX.shape

# %%
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# %%
testX.shape

# %%
trainX.shape

# %%
trainY.shape

# %%
trainX[12],testX[10]

# %%
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
def model_builder(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('input_unit',min_value=32,max_value=512,step=32), return_sequences=True, input_shape= ( trainX.shape[1], trainX.shape[2]), bias_regularizer = L1L2(0.009, 0.004)))
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

tuner.search(x=trainX, y=trainY, epochs = 200, batch_size =512, validation_data=(testX, testY), callbacks=[callbacks],shuffle = True)


# %%
tuner.results_summary()

# %%
best_model_S = tuner.get_best_models(num_models=1)[0]

# %%
history = best_model_S.fit(x=trainX, y=trainY, epochs = 150, batch_size =128, validation_data=(testX, testY), shuffle=False, verbose=0)

# %%
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# %%
predicted_silver_price = best_model_S.predict(testX)

# %%
predicted_silver_price = scaler_test.inverse_transform(predicted_silver_price.reshape(-1, 1))

# %%
true = scaler_test.inverse_transform(testY.reshape(-1, 1))

# %%
predicted_silver_price.shape

# %%
plt.plot(predicted_silver_price, label='predict')
plt.plot(true, label='true')
plt.legend()
plt.show()

# %%
# evaluate the model
scores = best_model_S.evaluate(trainX, trainY, verbose=0)
print("%s: %.2f%%" % (best_model.metrics_names[1], scores[1]*100))

# %%
best_model_S.save("models/silverPriceModel.h5")
print("Saved model to disk")

# %%
# load json and create model
# load model
model_s = load_model("models/silverPriceModel.h5", custom_objects={'root_mean_squared_error':                   
root_mean_squared_error})
print("dd")
# summarize model.
model_s.summary()
score = model_s.evaluate(trainX, trainY, verbose=0)
print("%s: %.2f%%" % (model_s.metrics_names[1], score[1]*100))

# %% [markdown]
# # Crude Oil Prices Forecasting Using LSTM

# %%
df_Crude=df[['Date','Crude Oil']]
df_Crude.head()

# %%
df_Crude=df_Crude.dropna()

# %%
prediction_days = 500
df_train_Cr= df_Crude['Crude Oil'][:len(df_Crude['Crude Oil'])-prediction_days].values.reshape(-1,1)
df_test_Cr= df_Crude['Crude Oil'][len(df_Crude['Crude Oil'])-prediction_days:].values.reshape(-1,1)

scaler_train = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler_train.fit_transform(df_train_Cr)

scaler_test = MinMaxScaler(feature_range=(0, 1))
scaled_test = scaler_test.fit_transform(df_test_Cr)

# %%
def create_dataset(dataset, look_back=30):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(scaled_train)
testX, testY = create_dataset(scaled_test)

# %%
trainX.shape,trainY.shape


# %%
testX.shape

# %%
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# %%
testX.shape,testY.shape

# %%
trainX.shape,trainY.shape

# %%
trainX[12],testX[10]

# %%
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
def model_builder(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('input_unit',min_value=32,max_value=512,step=32), return_sequences=True, input_shape= ( trainX.shape[1], trainX.shape[2]), bias_regularizer = L1L2(0.009, 0.004)))
    for i in range(hp.Int('n_layers', 1, 4)):
        model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=512,step=32),return_sequences=True))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.05)))
    model.add(Dense(1, activation=hp.Choice('dense_activation',values=['relu', 'sigmoid'],default='relu')))
   
    model.compile(loss=root_mean_squared_error, optimizer='adam',metrics = ['mse'])
    
    return model

tuner = kt.RandomSearch(model_builder, objective='val_loss', max_trials = 10, executions_per_trial =1,directory = "./Training/Crude/")

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', # Monitor the validation loss
                                                     min_delta=0,    # until it doesn't change (or gets worse)
                                                     patience=5,  # patience > 1 so it continutes if it is not consistently improving
                                                     verbose=0, 
                                                     mode='auto')]

tuner.search(x=trainX, y=trainY, epochs = 200, batch_size =512, validation_data=(testX, testY), callbacks=[callbacks],shuffle = True)


# %%
tuner.results_summary()

# %%
best_model_Cr = tuner.get_best_models(num_models=1)[0]

# %%
history = best_model_Cr.fit(x=trainX, y=trainY, epochs = 150, batch_size =128, validation_data=(testX, testY), shuffle=False, verbose=0)

# %%
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# %%
predicted_crude_price = best_model_Cr.predict(testX)

# %%
predicted_crude_price = scaler_test.inverse_transform(predicted_crude_price.reshape(-1, 1))

# %%
true = scaler_test.inverse_transform(testY.reshape(-1, 1))

# %%
plt.plot(predicted_crude_price, label='predict')
plt.plot(true, label='true')
plt.legend()
plt.show()

# %%
# evaluate the model
scores = best_model_Cr.evaluate(trainX, trainY, verbose=0)
print("%s: %.2f%%" % (best_model.metrics_names[1], scores[1]*100))

# %%
best_model_Cr.save("models/crudePriceModel.h5")
print("Saved model to disk")

# %%
# load json and create model
# load model
model_cr = load_model("models/crudePriceModel.h5", custom_objects={'root_mean_squared_error':                   
root_mean_squared_error})
print("dd")
# summarize model.
model_cr.summary()
score = model_cr.evaluate(trainX, trainY, verbose=0)
print("%s: %.2f%%" % (model_cr.metrics_names[1], score[1]*100))

# %% [markdown]
# # Brent Oil Prices Forecasting Using LSTM
#  

# %%
df_Brent=df[['Date','Brent Oil']]
df_Brent.head()

# %%
df_Brent=df_Brent.dropna()

# %%
df_Brent.info()

# %%
prediction_days = 500
df_train_B= df_Brent['Brent Oil'][:len(df_Brent['Brent Oil'])-prediction_days].values.reshape(-1,1)
df_test_B= df_Brent['Brent Oil'][len(df_Brent['Brent Oil'])-prediction_days:].values.reshape(-1,1)

scaler_train = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler_train.fit_transform(df_train_B)

scaler_test = MinMaxScaler(feature_range=(0, 1))
scaled_test = scaler_test.fit_transform(df_test_B)

trainX, trainY = create_dataset(scaled_train)
testX, testY = create_dataset(scaled_test)
trainX.shape,trainY.shape,testX.shape,testY.shape

# %%
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
trainX.shape,trainY.shape,testX.shape,testY.shape

# %%
trainX[12],testX[10]

# %%
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
def model_builder(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('input_unit',min_value=32,max_value=512,step=32), return_sequences=True, input_shape= ( trainX.shape[1], trainX.shape[2]), bias_regularizer = L1L2(0.009, 0.004)))
    for i in range(hp.Int('n_layers', 1, 4)):
        model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=512,step=32),return_sequences=True))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.05)))
    model.add(Dense(1, activation=hp.Choice('dense_activation',values=['relu', 'sigmoid'],default='relu')))
   
    model.compile(loss=root_mean_squared_error, optimizer='adam',metrics = ['mse'])
    
    return model

tuner = kt.RandomSearch(model_builder, objective='val_loss', max_trials = 10, executions_per_trial =1,directory = "./Training/Brent/")

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', # Monitor the validation loss
                                                     min_delta=0,    # until it doesn't change (or gets worse)
                                                     patience=5,  # patience > 1 so it continutes if it is not consistently improving
                                                     verbose=0, 
                                                     mode='auto')]

tuner.search(x=trainX, y=trainY, epochs = 200, batch_size =512, validation_data=(testX, testY), callbacks=[callbacks],shuffle = True)


# %%
tuner.results_summary()

# %%
best_model_Br = tuner.get_best_models(num_models=1)[0]

# %%
history = best_model_Br.fit(x=trainX, y=trainY, epochs = 150, batch_size =128, validation_data=(testX, testY), shuffle=False, verbose=0)

# %%
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# %%
predicted_brent_price = best_model_Br.predict(testX)

# %%
predicted_brent_price = scaler_test.inverse_transform(predicted_brent_price.reshape(-1, 1))

# %%
true = scaler_test.inverse_transform(testY.reshape(-1, 1))

# %%
plt.plot(predicted_brent_price, label='predict')
plt.plot(true, label='true')
plt.legend()
plt.show()

# %%
# evaluate the model
scores = best_model_Br.evaluate(trainX, trainY, verbose=0)
print("%s: %.2f%%" % (best_model.metrics_names[1], scores[1]*100))

# %%
best_model_Br.save("models/brentPriceModel.h5")
print("Saved model to disk")

# %%
# load json and create model
# load model
model_br = load_model("models/brentPriceModel.h5", custom_objects={'root_mean_squared_error':                   
root_mean_squared_error})
print("dd")
# summarize model.
model_br.summary()
score = model_br.evaluate(trainX, trainY, verbose=0)
print("%s: %.2f%%" % (model_br.metrics_names[1], score[1]*100))

# %% [markdown]
# # Natural Gas Prices Forecasting Using LSTM
# 

# %%
df_Natural=df[['Date','Natural Gas']]
df_Natural.head()

# %%
df_Natural=df_Natural.dropna()

# %%
df_Natural.info()

# %%
prediction_days = 500
df_train_N= df_Natural['Natural Gas'][:len(df_Natural['Natural Gas'])-prediction_days].values.reshape(-1,1)
df_test_N= df_Natural['Natural Gas'][len(df_Natural['Natural Gas'])-prediction_days:].values.reshape(-1,1)

scaler_train = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler_train.fit_transform(df_train_N)

scaler_test = MinMaxScaler(feature_range=(0, 1))
scaled_test = scaler_test.fit_transform(df_test_N)

trainX, trainY = create_dataset(scaled_train)
testX, testY = create_dataset(scaled_test)

trainX.shape,trainY.shape,testX.shape,testY.shape

# %%
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
trainX.shape,trainY.shape,testX.shape,testY.shape

# %%
trainX[12],testX[10]

# %%
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
def model_builder(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('input_unit',min_value=32,max_value=512,step=32), return_sequences=True, input_shape= ( trainX.shape[1], trainX.shape[2]), bias_regularizer = L1L2(0.009, 0.004)))
    for i in range(hp.Int('n_layers', 1, 4)):
        model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=512,step=32),return_sequences=True))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.05)))
    model.add(Dense(1, activation=hp.Choice('dense_activation',values=['relu', 'sigmoid'],default='relu')))
   
    model.compile(loss=root_mean_squared_error, optimizer='adam',metrics = ['mse'])
    
    return model

tuner = kt.RandomSearch(model_builder, objective='val_loss', max_trials = 10, executions_per_trial =1,directory = "./Training/Natural/")

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', # Monitor the validation loss
                                                     min_delta=0,    # until it doesn't change (or gets worse)
                                                     patience=5,  # patience > 1 so it continutes if it is not consistently improving
                                                     verbose=0, 
                                                     mode='auto')]

tuner.search(x=trainX, y=trainY, epochs = 200, batch_size =512, validation_data=(testX, testY), callbacks=[callbacks],shuffle = True)


# %%
tuner.results_summary()

# %%
best_model_N = tuner.get_best_models(num_models=1)[0]

# %%
history = best_model_N.fit(x=trainX, y=trainY, epochs = 150, batch_size =128, validation_data=(testX, testY), shuffle=False, verbose=0)

# %%
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# %%
predicted_naturalGas_price = best_model_N.predict(testX)

# %%
predicted_naturalGas_price = scaler_test.inverse_transform(predicted_naturalGas_price.reshape(-1, 1))

# %%
true = scaler_test.inverse_transform(testY.reshape(-1, 1))

# %%
plt.plot(predicted_naturalGas_price, label='predict')
plt.plot(true, label='true')
plt.legend()
plt.show()

# %%
# evaluate the model
scores = best_model_N.evaluate(trainX, trainY, verbose=0)
print("%s: %.2f%%" % (best_model.metrics_names[1], scores[1]*100))

# %%
best_model_N.save("models/naturalGasPriceModel.h5")
print("Saved model to disk")

# %%
# load json and create model
# load model
model_N = load_model("models/naturalGasPriceModel.h5", custom_objects={'root_mean_squared_error':                   
root_mean_squared_error})
print("dd")
# summarize model.
model_N.summary()
score = model_N.evaluate(trainX, trainY, verbose=0)
print("%s: %.2f%%" % (model_br.metrics_names[1], score[1]*100))

# %% [markdown]
# # Copper Prices Forecasting Using LSTM
# 

# %%
df_Copper=df[['Date','Copper']]
df_Copper.head()

# %%
df_Copper=df_Copper.dropna()

# %%
df_Copper.info()

# %%
prediction_days = 500
df_train_Co= df_Copper['Copper'][:len(df_Copper['Copper'])-prediction_days].values.reshape(-1,1)
df_test_Co= df_Copper['Copper'][len(df_Copper['Copper'])-prediction_days:].values.reshape(-1,1)

scaler_train = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler_train.fit_transform(df_train_N)

scaler_test = MinMaxScaler(feature_range=(0, 1))
scaled_test = scaler_test.fit_transform(df_test_N)

trainX, trainY = create_dataset(scaled_train)
testX, testY = create_dataset(scaled_test)

trainX.shape,trainY.shape,testX.shape,testY.shape

# %%
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
trainX.shape,trainY.shape,testX.shape,testY.shape

# %%
trainX[12],testX[10]

# %%
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
def model_builder(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('input_unit',min_value=32,max_value=512,step=32), return_sequences=True, input_shape= ( trainX.shape[1], trainX.shape[2]), bias_regularizer = L1L2(0.009, 0.004)))
    for i in range(hp.Int('n_layers', 1, 4)):
        model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=512,step=32),return_sequences=True))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.05)))
    model.add(Dense(1, activation=hp.Choice('dense_activation',values=['relu', 'sigmoid'],default='relu')))
   
    model.compile(loss=root_mean_squared_error, optimizer='adam',metrics = ['mse'])
    
    return model

tuner = kt.RandomSearch(model_builder, objective='val_loss', max_trials = 10, executions_per_trial =1,directory = "./Training/Copper/")

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', # Monitor the validation loss
                                                     min_delta=0,    # until it doesn't change (or gets worse)
                                                     patience=5,  # patience > 1 so it continutes if it is not consistently improving
                                                     verbose=0, 
                                                     mode='auto')]

tuner.search(x=trainX, y=trainY, epochs = 200, batch_size =512, validation_data=(testX, testY), callbacks=[callbacks],shuffle = True)


# %%
tuner.results_summary()

# %%
best_model_Co = tuner.get_best_models(num_models=1)[0]

# %%
history = best_model_Co.fit(x=trainX, y=trainY, epochs = 150, batch_size =128, validation_data=(testX, testY), shuffle=False, verbose=0)

# %%
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# %%
predicted_copper_price = best_model_Co.predict(testX)

# %%
predicted_copper_price = scaler_test.inverse_transform(predicted_copper_price.reshape(-1, 1))

# %%
true = scaler_test.inverse_transform(testY.reshape(-1, 1))

# %%
plt.plot(predicted_copper_price, label='predict')
plt.plot(true, label='true')
plt.legend()
plt.show()

# %%
# evaluate the model
scores = best_model_Co.evaluate(trainX, trainY, verbose=0)
print("%s: %.2f%%" % (best_model.metrics_names[1], scores[1]*100))

# %%
best_model_Co.save("models/copperPriceModel.h5")
print("Saved model to disk")

# %%
# load json and create model
# load model
model_Co = load_model("models/copperPriceModel.h5", custom_objects={'root_mean_squared_error':                   
root_mean_squared_error})
print("dd")
# summarize model.
model_Co.summary()
score = model_Co.evaluate(trainX, trainY, verbose=0)
print("%s: %.2f%%" % (model_Co.metrics_names[1], score[1]*100))

# %% [markdown]
#  # Results
#  As we can see we build our model from the data we extracted from [Investing.com](https://www.investing.com/) from last 10 years and performed EDA on it and we could see how variuos commodities are related to each other then build LSTM models for each of the commodities
#  and here are the **MSE** values for each of them
# ### **Gold**  -> **0.20** 
# ### **Silver** -> **0.05** 
# ### **Crude  Oil** -> **0.47** 
# ### **Brent Oil** -> **0.17**
# ### **Natural Gas** -> **0.05**
# ### **Copper** -> **0.04**


