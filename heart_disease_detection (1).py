# -*- coding: utf-8 -*-
"""Heart_disease_detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rZcqH9XRoBReeS6DguMTly2ASu8FBTBf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

hd = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/heart.csv')

hd.head(5)

sns.pairplot(hd)

sns.countplot(x = 'target', data = hd)

sns.pairplot(hd, hue = 'target', palette = 'deep')

selected_features = ['age',	'sex',	'cp',	'trestbps',	'chol',	'fbs',	'restecg',	'thalach',	'exang',	'oldpeak',	'slope',	'ca',	'thal']

X = hd[selected_features]
y = hd['target']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled

X_train,X_test,y_train,y_test = train_test_split(X_scaled, y, test_size = 0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

predictor = tf.keras.models.Sequential()
predictor.add(tf.keras.layers.Dense(units = 100, activation = 'relu', input_shape = (13,)))
predictor.add(tf.keras.layers.Dropout(0.3))
predictor.add(tf.keras.layers.Dense(units = 50, activation = 'relu'))
predictor.add(tf.keras.layers.Dropout(0.3))
predictor.add(tf.keras.layers.Dense(units = 50, activation = 'relu'))
predictor.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

predictor.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = 'accuracy')

epochs_hist = predictor.fit(X_train,y_train,epochs = 50, batch_size = 125)

epochs_hist2 = predictor.fit(X_test,y_test,epochs = 50, batch_size = 125)

evaluation = predictor.evaluate(X_test,y_test)
print('test accuracy:{}'.format(evaluation[1]))

eh = epochs_hist.history['accuracy']
eh2 = epochs_hist.history['loss']
plt.plot(eh)
plt.plot(eh2)
plt.title('Training loss and Accuracy vs epoch graphs')
plt.xlabel('epochs')
plt.ylabel('Training_loss and Accuracy')
plt.legend({'Accuracy', 'loss'})

eh3 = epochs_hist2.history['accuracy']
eh4 = epochs_hist2.history['loss']
plt.plot(eh3)
plt.plot(eh4)
plt.title('Testing loss and Accuracy vs epoch graphs')
plt.xlabel('epochs')
plt.ylabel('Testing_loss and Accuracy')
plt.legend({'Accuracy', 'loss'})

y_predict = predictor.predict(X_test)

y_predict

y_predict = (y_predict > 0.5)

y_predict

from sklearn.metrics import confusion_matrix
y_train_pred = predictor.predict(X_train)
y_train_pred = (y_train_pred > 0.5)
cm = confusion_matrix(y_train, y_train_pred)
sns.heatmap(cm, annot = True)

from sklearn.metrics import classification_report
print(classification_report(y_train_pred, y_train))

cm2 = confusion_matrix(y_test, y_predict)
sns.heatmap(cm2, annot = True)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))