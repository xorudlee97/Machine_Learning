from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import img_to_array, array_to_img

import matplotlib.pyplot as plt
from keras import regularizers
import numpy as np

# CIFAR_10은 3채널로 구성된 32*32 이미지 60000장을 갖는다.
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

# 상수의 정의
BATCH_SIZE = 128
NB_EPOCH = 100
# Eearly_Stop = 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

# 데이터 셋 불러 오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 범주형으로 변환
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3).astype('float32') / 255
# reshape 28 => 48 증폭
x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_train])
x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_test])

from keras.applications import VGG16
conv_base = VGG16(weights='imagenet', include_top=False,
                  input_shape=(48,48,3))


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
                       batch_size=BATCH_SIZE, verbose=VERBOSE)
print(acc)
