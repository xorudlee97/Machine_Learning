import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import np_utils
import os

file_dir = os.path.dirname("D:/LTK_AI_Study/Machine_Learning/Data/")
# 데이터 읽기
wine = pd.read_csv(file_dir+"/winequality-white.csv", sep=";", encoding='utf-8')
# 데이터 불리 하기
y = wine["quality"]
x = wine.drop("quality", axis= 1)


newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

y = np_utils.to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

# 학습하기

from keras.models import Sequential
from keras.layers import Dense,Dropout

model = Sequential()
model.add(Dense(128, input_shape=(11,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=500, batch_size=10, validation_data=(x_val, y_val))
loss, acc = model.evaluate(x_test, y_test, batch_size=3)
# y_pridect = model.predict(x_test)
print("정답률=",acc)
# print(y_pridect)
'''
머신 러닝과 딥러닝의 모델링 구조는 차이가 없다.
머신 러닝
model = RandomFrest()
model.fit(x_train, y_train)
acc = model.scroe(x_test, y_test)
y_pred = model.predict(x_test)

딥러닝
model = Sequenecial()
model.fit(x_train, y_train)
loss, acc = model.eveludate(*)
'''
'''
Standard_Scaler
model.add(Dense(128, input_shape=(11,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
epochs=300, batch_size=10
0.64
'''