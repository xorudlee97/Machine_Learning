from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy
import tensorflow as tf

import os

scaler = StandardScaler()
# scaler = MinMaxScaler()

file_dir = os.path.dirname("D:/LTK_AI_Study/Machine_Learning/Data/")

# seed 값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = numpy.loadtxt(file_dir+"/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
scaler.fit(X)
X = scaler.transform(X)
Y = dataset[:,8]

#모델의 실행
model = Sequential()
model.add(Dense(128, input_shape = (8,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 모델 실행
model.fit(X,Y, epochs=200, batch_size=10)

# 결과 출력
print("\n Acc: %.4f" %(model.evaluate(X, Y)[1]))

'''
실습 1 케라스모델을 머신러닝 모델로 변경
실습 2 
'''