# -*- coding: utf-8 -*-
"""Maruti Stock Price Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gsNhseF2f0QMkZMtWhdMViREHqYkq2Vw
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

maruti = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/MARUTI.csv")

maruti.head(5)

maruti.drop(labels = {'Trades'}, axis = 1,inplace = True)

maruti.head(5)

sns.pairplot(maruti)

maruti.info()

maruti['Deliverable Volume'] = maruti['Deliverable Volume'].fillna('ffill')
maruti['%Deliverble'] = maruti['%Deliverble'].fillna('ffill')

maruti['Profit'] = maruti['Close'] - maruti['Open']

maruti.info()

X = maruti[['VWAP','Open', 'High', 'Low', 'Prev Close', 'Volume', 'Profit']]

y = maruti[['Close']]

X

y

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_scaled,test_size = 0.2)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(X_train,y_train)
print("Training_Score:",model1.score(X_train, y_train)*100)
print("Testing_score :",model1.score(X_test, y_test)*100)

y_predict = model1.predict(X_test)

plt.plot(y_test, y_predict, '^', color = 'r')
plt.xlabel('y_test')
plt.ylabel('y_predict')

y_predict_original = scaler.inverse_transform(y_predict)
y_test_original = scaler.inverse_transform(y_test)
plt.plot(y_test_original,y_predict_original, '^', color = 'b')

k = X_test.shape
n = len(X_test)
print('value of n:',n)

RMSE = float(format(np.sqrt(mean_squared_error(y_test_original, y_predict_original)), '0.3f'))
RMSE

MSE = mean_squared_error(y_test_original,y_predict_original)
print("Mean_squared_error:",MSE)

MAE = mean_absolute_error(y_test_original,y_predict_original)
print("Mean_absolute_error:",MAE)

r2 = r2_score(y_test_original,y_predict_original)
print("r2_score:",r2)