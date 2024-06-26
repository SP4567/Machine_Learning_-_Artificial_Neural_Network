# -*- coding: utf-8 -*-
"""Bike_Rental_Usage.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WEPUFS4P61DQe4SNl8woeZ9L6cb860hE
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

bike  = pd.read_csv('/content/bike_sharing_daily.csv')
bike

bike.head()

bike.info()

bike.describe()

sns.heatmap(bike.isnull())

bike  = bike.drop(labels = {'instant'}, axis = 1)

bike

bike.dteday = pd.to_datetime(bike.dteday, format = '%m/%d/%Y')

bike

bike.index = pd.DatetimeIndex(bike.dteday)

bike

bike = bike.drop(labels = {'dteday'}, axis = 1)

bike

bike = bike.drop({'casual', 'registered'}, axis = 1)

bike

bike['cnt'].asfreq('w').plot(linewidth = 3)
plt.title("Bike Rental Usage Per Week")
plt.xlabel('Week')
plt.ylabel('Bike Rental')

bike['cnt'].asfreq('M').plot(linewidth = 3)
plt.title('Bke Rental Usage Per Month')
plt.ylabel('Bike Rental')
plt.xlabel('Month')

bike['cnt'].asfreq('Q').plot(linewidth = 3)
plt.title('Bke Rental Usage Per Month')
plt.ylabel('Bike Rental')
plt.xlabel('Month')

sns.pairplot(bike)

X_numerical = bike[['temp', 'hum', 'windspeed', 'cnt']]

X_numerical

sns.pairplot(X_numerical)

sns.heatmap(X_numerical.corr(), annot = True)

X_cat = bike[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']]

X_cat

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()

X_cat

X_cat.shape

X_cat = pd.DataFrame(X_cat)

X_cat

X_numerical = X_numerical.reset_index()
X_numerical

X_all = pd.concat([X_cat,X_numerical],axis = 1)

X_all

X_all = X_all.drop('dteday',axis = 1)
X_all

X = X_all.iloc[:, :-1].values
y = X_all.iloc[:, -1:].values

X.shape

y.shape

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
y = scalar.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, y, test_size = 0.2)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units = 100, activation = 'tanh', input_shape = (35,)))
model.add(tf.keras.layers.Dense(units = 100, activation = 'tanh'))
model.add(tf.keras.layers.Dense(units = 100, activation = 'tanh'))
model.add(tf.keras.layers.Dense(units = 1, activation = 'linear'))

model.summary()

model.compile(optimizer = 'Adam', loss = 'mean_squared_error')

epoch_hist = model.fit(X_train, Y_train, epochs = 25, batch_size = 50, validation_split = 0.2 )

epoch_hist.history.keys()
eh = epoch_hist.history['loss']
eh2 = epoch_hist.history['val_loss']
plt.plot(eh)
plt.plot(eh2)
plt.title('Loss dring the training')
plt.xlabel('epochs')
plt.ylabel('Training and validation_loss')
plt.legend(['training loss','validation loss'])

y_predict = model.predict(X_test)
plt.plot(Y_test,y_predict,'^',color = 'r')
plt.xlabel('model predictions')
plt.ylabel('true values')

y_predict_original = scalar.inverse_transform(y_predict)

y_test_original = scalar.inverse_transform(Y_test)

plt.plot(y_test_original,y_predict_original,'^',color = 'r')
plt.xlabel('model predictions')
plt.ylabel('true values')

k = X_test.shape

k

n = len(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
RMSE = float(format(np.sqrt(mean_squared_error(y_test_original,y_predict_original)), '0.3f'))

print(RMSE)

MSE = mean_squared_error(y_test_original,y_predict_original)
MAE = mean_absolute_error(y_test_original, y_predict_original)
r2 = r2_score(y_test_original, y_predict_original)

MSE
MAE
r2
print("MSE = ",MSE,"\nMAE = ",MAE,"\nR2 = ",r2)

adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)