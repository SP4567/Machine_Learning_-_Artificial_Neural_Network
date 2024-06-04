# -*- coding: utf-8 -*-
"""Crop_Production.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MaB16FGJXs6AL24VYg43TKFu4KgJHTQp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('/content/drive/MyDrive/Crop Production data.csv')

df.head(10)

df.info()

mean_production = df['Production'].mean()

mean_production

df['Production'] = df['Production'].fillna(value = mean_production)

df.info()

sns.countplot(x = 'Season', data = df)

from matplotlib import pyplot as plt
import seaborn as sns
df.groupby('Season').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

def _plot_series(series, series_name, series_index=0):
  palette = list(sns.palettes.mpl_palette('Dark2'))
  xs = series['Crop_Year']
  ys = series['Area']

  plt.plot(xs, ys, label=series_name, color=palette[series_index % len(palette)])

fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
df_sorted = df.sort_values('Crop_Year', ascending=True)
for i, (series_name, series) in enumerate(df_sorted.groupby('Season')):
  _plot_series(series, series_name, i)
  fig.legend(title='Season', bbox_to_anchor=(1, 1), loc='upper left')
sns.despine(fig=fig, ax=ax)
plt.xlabel('Crop_Year')
_ = plt.ylabel('Area')

df['Area'].plot(kind='line', figsize=(8, 4), title='Area')
plt.gca().spines[['top', 'right']].set_visible(False)

df['Area'].plot(kind='hist', bins=20, title='Area')
plt.gca().spines[['top', 'right',]].set_visible(False)

df.head(15)

print("Total Production:", df.Production.sum())

print("Total Area:", df.Area.sum())

df.shape

df['District_Name'].value_counts()

df['State_Name'].value_counts()

df['Season'].value_counts()

new_State = pd.get_dummies(df['State_Name'], dtype = 'int')
new_District = pd.get_dummies(df['District_Name'], dtype = 'int')
new_Season = pd.get_dummies(df['Season'], dtype = 'int')
new_Crop = pd.get_dummies(df['Crop'], dtype = 'int')

State_ = pd.DataFrame(new_State)
District_= pd.DataFrame(new_District)
Season_ = pd.DataFrame(new_Season)
Crop_ = pd.DataFrame(new_Crop)

new_df = pd.concat([State_, District_, Season_, Crop_, df], axis = 1)

new_df.head(10)

new_df = new_df.drop({'State_Name', 'District_Name', 'Season', 'Crop'}, axis = 1)

new_df.head(10)

X = new_df.drop('Production', axis = 1)
y  =new_df[['Production']]

print(X.shape)
print(y.shape)

Scaler = MinMaxScaler()
X = Scaler.fit_transform(X)
y = Scaler.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units = 128, activation = 'relu', input_shape = (811,)))
model.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units = 32, activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 32, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units = 16, activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 16, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units = 8, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units = 4, activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 1, activation  = 'linear'))
model.summary()

keras.utils.plot_model(model, to_file='png', show_shapes=True)

model.compile(optimizer = 'Adam', loss = 'mean_squared_error')
from keras.callbacks import EarlyStopping
es = EarlyStopping(patience = 2, monitor = 'val_loss')
model.fit(X_train, y_train, epochs = 25, batch_size = 10, validation_data = (X_test, y_test), callbacks = [es])

hist = model.history.history
h = pd.DataFrame(hist)
h.plot()

y_predict = model.predict(X_test)
y_predict

plt.plot(y_test,y_predict, '^', color = 'r')
plt.xlabel('y_test')
plt.ylabel('y_predict')

y_predict_original = Scaler.inverse_transform(y_predict)
y_test_original = Scaler.inverse_transform(y_test)
plt.plot(y_test_original,y_predict_original,'^',color = 'b')
plt.xlabel('model_predictions')
plt.ylabel('true_values')

k = X_test.shape
k
n = len(X_test)
print('\n')
n

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from math import sqrt
RMSE = float(format(np.sqrt(mean_squared_error(y_test_original,y_predict_original)), '0.3f'))
print(RMSE)

MSE = mean_squared_error(y_test_original,y_predict_original)
print(MSE)

MAE = mean_absolute_error(y_test_original,y_predict_original)
print(MAE)

r2 = r2_score(y_test_original,y_predict_original)
print(r2)