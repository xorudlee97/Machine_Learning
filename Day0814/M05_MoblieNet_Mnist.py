import numpy as np
import pandas as pd

from keras.utils import np_utils
from keras.datasets import mnist
from keras.preprocessing.image import img_to_array, array_to_img
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)
x_train=np.dstack([x_train] * 3)
x_test=np.dstack([x_test] * 3)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 3).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 3).astype('float32') / 255
# reshape 28 => 48 증폭
x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((75,75))) for im in x_train])
x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((75,75))) for im in x_test])
print(x_train.shape)
print(x_test.shape)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

from keras.applications import MobileNet
conv_base = MobileNet(weights='imagenet', include_top=False,
                  input_shape=(75,75,3))


from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=2000, verbose=1)

loss, acc = model.evaluate(x_test, y_test, 
                       batch_size=2000, verbose=1)
print(acc)
