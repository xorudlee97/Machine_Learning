from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
import numpy as np

# 데이터 
# XOR 모델
#   0 1
# 0 0 1
# 1 1 0
# 한개의 그래프로는 0과 1을 구분 지을 수가 없다.

x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]
x_data = np.array(x_data)
y_data = np.array(y_data)

model = Sequential()
model.add(Dense(12, input_shape = (2,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 실행
# XOR 검출 binary_crossentropy
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_data, y_data, epochs=100, batch_size=1)

# 예측

x_test = [[0,0],[1,0],[0,1],[1,1]]
x_test = np.array(x_test)
loss, acc = model.evaluate(x_data,y_data)
y_predict = model.predict_classes(x_test)

print(y_predict)
print("acc = ", acc)
'''
# 모델
# model = LinearSVC()
model = SVC()

# 실행
model.fit(x_data, y_data)

# 예측

x_test = [[0,0],[1,0],[0,1],[1,1]]
y_predict = model.predict(x_test)

print(x_test, "의 예측 결과", y_predict)
print("acc = ", accuracy_score([0,1,1,0], y_predict))
'''