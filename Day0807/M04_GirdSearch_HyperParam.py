#-*- coding: utf-8 -*-

# train Data 60000
# test Data 10000

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy
import os
import tensorflow as tf

# 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# X_train = X_train.reshape(X_train.shape[0], 28,28,1).astype('float32') / 255
# X_test = X_test.reshape(X_test.shape[0], 28,28,1).astype('float32') / 255
X_train = X_train.reshape(X_train.shape[0], 28 * 28).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28 * 28).astype('float32') / 255

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)


from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np

def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28,),name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(10, activation='softmax', name = 'output')(x)
    model = Model(inputs=inputs, outputs = prediction)
    model.compile(optimizer=optimizer, loss= 'categorical_crossentropy',metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size":batches, "optimizer":optimizers, "keep_prob":dropout}

from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn= build_network, verbose=1)
hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, KFold

kflod_cv = KFold(n_splits=5, shuffle=True)
search = GridSearchCV(model, hyperparameters, cv=kflod_cv)
search.fit(X_train, Y_train)
print(search.best_params_)
'''
머신러닝을 실행 중 조정 가능한 
파라미터에서 가장 좋은 값을 찾아내기 위해 
파라미터 조정을 하는 프로그래밍
리턴 가장 최적의 파라미터를 리턴
'''

# CNN 모델로 바꿀것