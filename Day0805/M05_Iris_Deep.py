import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from keras.utils import np_utils
import numpy as np
import os

file_dir = os.path.dirname("D:/LTK_AI_Study/Machine_Learning/Data/")
iris_data = pd.read_csv(file_dir+"/iris.csv", encoding='utf-8', names=['SepalLength','SepalWidth','PataLength','PatalWidth','Name'])
# print(iris_data)
# print(iris_data.shape)
# print(type(iris_data))

# 레이블로 자르기
y = iris_data.loc[:, 'Name']
x = iris_data.loc[:, ['SepalLength', 'SepalWidth', 'PataLength', 'PatalWidth']]

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# 라벨 인코더 변환 후 원 핫 인코딩
change_param_title = ['Iris-setosa','Iris-versicolor','Iris-virginica']
lable = LabelEncoder()
lable.fit(change_param_title)
y = lable.transform(y)
# 원핫 인코딩 변환
y = np_utils.to_categorical(y)
# print(y)
# print(y)
# 열로 자르기
# x = iris_data.iloc[:, 4]
# y = iris_data.iloc[:,0:4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size = 0.8, shuffle=True)
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# 학습 하기

model = Sequential()
model.add(Dense(128, input_shape = (4,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=200, batch_size=10)
loss, acc = model.evaluate(x_test, y_test)
y_pridect = model.predict(x_test)
y_pridect = np.round(y_pridect).astype(int)
y_pridect = np.argmax(y_pridect, axis=1)
y_pridect = lable.inverse_transform(y_pridect)
print(acc)
print(y_pridect)