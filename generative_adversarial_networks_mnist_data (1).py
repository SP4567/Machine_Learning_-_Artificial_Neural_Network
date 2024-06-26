# -*- coding: utf-8 -*-
"""GENERATIVE_ADVERSARIAL_NETWORKS_Mnist_Data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BHAcJC2D1deN_rxLOxJL8w2BdWR8CFAK
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.models import Sequential

(X_train,y_train),(X_test,y_test) = mnist.load_data()

plt.imshow(X_train[0])

y_train==0

only_zeros = X_train[y_train==0]

only_zeros.shape

X_train.shape

plt.imshow(only_zeros[14])

coding_size = 200
generator = Sequential()
generator.add(Dense(100, activation = 'relu', input_shape = [coding_size]))
generator.add(Dense(150, activation = 'relu'))
generator.add(Dense(784, activation = 'relu'))
generator.add(Reshape([28,28]))

discriminator = Sequential()
discriminator.add(Flatten(input_shape = [28,28]))
discriminator.add(Dense(150, activation = 'relu'))
discriminator.add(Dense(100,activation = 'relu'))
discriminator.add(Dense(1, activation = 'sigmoid'))
discriminator.compile(loss = 'binary_crossentropy', optimizer = 'adam')

GAN = Sequential([generator, discriminator])
discriminator.trainable = False

GAN.compile(loss = 'binary_crossentropy', optimizer = 'adam')

batch_size = 32

my_data = only_zeros #my_data = X_train

dataset = tf.data.Dataset.from_tensor_slices(my_data).shuffle(buffer_size = 1000)

type(dataset)

dataset = dataset.batch(batch_size,drop_remainder=True).prefetch(1)

epochs = 55

GAN.layers[0].layers

GAN.layers[1].layers

GAN.layers[0].summary()

GAN.layers[1].summary()

generator, discriminator = GAN.layers
for epoch in range(epochs):
  print(f"Currently on Epoch{epoch+1}")
  i = 0;
  for X_batch in dataset:
    i = i+1
    if i%100 == 0:
      print(f"\t Currently on batch number {i} of {(len(my_data)//batch_size)}")

      #Discriminator
      noise = tf.random.normal(shape = [batch_size, coding_size])
      gen_images = generator(noise)
      X_fake_vs_real = tf.concat([gen_images, tf.dtypes.cast(X_batch, tf.float32)], axis = 0)
      y1 = tf.constant([[0.0]]*batch_size + [[1.0]]*batch_size)
      discriminator.trainable = True
      discriminator.train_on_batch(X_fake_vs_real,y1)

      #train generator
      noise = tf.random.normal(shape = [batch_size, coding_size])
      y2 = tf.constant([[1.0]]*batch_size)
      discriminator.trainable = False
      GAN.train_on_batch(noise, y2)

noise = tf.random.normal(shape = [10, coding_size])

noise.shape

plt.imshow(noise)

images = generator(noise)

images.shape

plt.imshow(images[0])

plt.imshow(images[1])

plt.imshow(images[2])

plt.imshow(images[4])

for image in images:
    plt.imshow(image.numpy().reshape(28,28))
    plt.show()