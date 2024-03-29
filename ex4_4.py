import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, Flatten, MaxPooling2D
model = Sequential()
N = 10  # Number of feature maps
w, h = 5, 5  # Conv. window size

model.add(Conv2D(filters=32, kernel_size=(w, h), input_shape=(64, 64, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(filters=32, kernel_size=(w, h), activation='relu', padding='same'))
model.add(MaxPooling2D((4, 4)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.summary()

