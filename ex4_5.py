from keras.models import Sequential
import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
import keras
from Dataset.GTSRB_subset.simplelbp import local_binary_pattern
from Dataset.GTSRB_subset.traffic_signs import load_data, extract_lbp_features
from sklearn.model_selection import train_test_split
model = Sequential()
N = 10  # Number of feature maps
w, h = 5, 5  # Conv. window size
X, Y = load_data("./Dataset/GTSRB_subset")
# F = extract_lbp_features(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# print(Y.shape)

x_train /= 255
x_test /= 255
x_train = np.reshape(x_train, (len(x_train), 64, 64, 1))
x_test = np.reshape(x_test, (len(x_test), 64, 64, 1))
# print(x_train.shape)
model.add(Conv2D(filters=32, kernel_size=(w, h), input_shape=(64, 64, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(filters=32, kernel_size=(w, h), activation='relu', padding='same'))
model.add(MaxPooling2D((4, 4)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=[x_test, y_test])
