from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 기온 데이터 읽어 들이기
file_dir = os.path.dirname("D:/LTK_AI_Study/Machine_Learning/Data/")
df = pd.read_csv(file_dir+"/tem10y.csv", sep=",", encoding='utf-8')

# 데이터를 학습전용과 테스트 전용으로 분리
train_year = (df["연"] <= 2015)
test_year = (df["연"] >= 2016)
interval = 6

# 과거 6일의 데이터를 기반으로 학습할 데이터 만들기
def make_data(data):
    x = [] # 학습 데이터
    y = [] # 경과
    temps = list(data["기온"])
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    x = np.array(x)
    y = np.array(y)
    return (x, y)

x_train, y_train = make_data(df[train_year])
x_test, y_test = make_data(df[test_year])
x_test = np.array(x_test)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train)

# 케라스 모델 분석
model = Sequential()
model.add(LSTM(128, input_shape=(6,1), return_sequences=True))
model.add(LSTM(64))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(16, activation= 'relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=300, validation_data=(x_val, y_val))
loss, acc = model.evaluate(x_test, y_test, batch_size=3)
y_predict = model.predict(x_test)
# print("정답률=",acc)
# RMSE 구하기
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
r2_y_predict = r2_score(y_test, y_predict)
print("R2   : ", r2_y_predict)

# 결과 그래프 그리기
# plt.figure(figsize=(10, 6), dpi=100)
# plt.plot(y_test, c="r")
# plt.plot(y_predict, c="b")
# plt.savefig('tenki-kion-lr.png')
# plt.show()