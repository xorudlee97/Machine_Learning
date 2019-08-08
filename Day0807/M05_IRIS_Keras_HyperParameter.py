import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
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
def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape = (4,), name= 'inputs')
    x = Dense(128, activation='relu', name='hidden1')(inputs)
    x = Dense(32, activation='relu', name='hidden2')(x)
    x = Dense(32, activation='relu', name='hidden3')(x)
    x = Dense(32, activation='relu', name='hidden4')(x)
    x = Dense(16, activation='relu', name='hidden5')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(3, activation='softmax', name = 'output')(x)
    model = Model(inputs=inputs, outputs = prediction)
    model.compile(optimizer=optimizer, loss= 'categorical_crossentropy',metrics=['accuracy'])
    return model
def create_hyperparameters():
    batches = [1,2,3,4,5,6,7,8,9,10]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.3, 5)
    return{"batch_size":batches, "optimizer":optimizers, "keep_prob":dropout}

from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn= build_network, verbose=1)
hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(estimator=model,
                            param_distributions=hyperparameters,
                            n_iter=10, n_jobs=1, cv=3, verbose=1)
search.fit(x_train, y_train)

model = build_network(search.best_params_["keep_prob"], search.best_params_['optimizer'])
model.fit(x_train,y_train, epochs=200, batch_size = search.best_params_["batch_size"])
loss, acc = model.evaluate(x_test, y_test)
y_pridect = model.predict(x_test)
y_pridect = np.round(y_pridect).astype(int)
y_pridect = np.argmax(y_pridect, axis=1)
y_pridect = lable.inverse_transform(y_pridect)
print(search.best_params_)
print(acc)
print(y_pridect)