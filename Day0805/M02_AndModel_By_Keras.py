from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np 

# 데이터 
# AND 모델
#   0 1
# 0 0 0
# 1 0 1
# =>[[0,0],[1,0],[0,1],[1,1]]
# [0,0,0,1]
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,0,0,1]

x_data = np.array(x_data)
y_data = np.array(y_data)
# print(x_data.shape)
# print(y_data.shape)

# x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1],1))
# y_data = np.reshape(y_data, (y_data.shape[0], 1))
# x_data = np.transpose(x_data)
# print(x_data)
# print(y_data)

# 모델
model = Sequential()

model.add(Dense(32,input_shape=(2,)))
model.add(Dense(16))
model.add(Dense(1, activation="sigmoid"))
# 실행
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_data, y_data, epochs=100, batch_size=1)

# 예측

x_test = [[0,0],[1,0],[0,1],[1,1]]
x_test = np.array(x_test)
loss, acc = model.evaluate(x_data,y_data)
y_predict = model.predict(x_test)

print(y_predict)
print("acc = ", acc)