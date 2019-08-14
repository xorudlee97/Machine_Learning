import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# csv 파일 받기
file_dir = os.path.dirname("D:/LTK_AI_Study/Test/")
csv_kospi = pd.read_csv(file_dir+"/kospi200test.csv", encoding='cp949')

csv_kospi = csv_kospi.to_numpy()
bat_size = 1
print(len(csv_kospi))
# 날짜 기준 파일 받기
def kospi_split_date(seq, number):
    arr = []
    for i in range(len(seq), number-1, -1):
        sub = seq[(i-number):i]
        sub = sub[::-1]
        arr.append([item for item in sub])
    return np.array(arr)
csv_kospi_list = kospi_split_date(csv_kospi, 6)


# 데이터 정렬
# 시가, 저가, 고가, 종가 추출
X = csv_kospi_list[:,:,1:4]
# 종가 추출
Y = csv_kospi_list[:,:,4:5]
# print(Y.shape)

x_predict = X[-1]
# print(x_predict)
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.3, random_state = 777
)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

##############모델 만들기###################

from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(LSTM(128, batch_input_shape=(bat_size, x_train.shape[1],3), stateful=True))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1, activation='relu'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=200, batch_size=5, verbose=1)

_, acc = model.evaluate(x_test, y_test, batch_size = bat_size)
Kospi_Y_predict = model.predict(x_predict)
print(Kospi_Y_predict)