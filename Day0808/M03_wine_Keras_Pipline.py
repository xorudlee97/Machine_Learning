import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import np_utils
import numpy as np
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
# scaler = StandardScaler()
# scaler.fit(x)
# x = scaler.transform(x)

y = np_utils.to_categorical(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

# 학습하기

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input

model = Sequential()
def build_Model(keep_prob=0.3, optimizer='adam'):
    inputs = Input(shape=(11,), name='inputs')
    x = Dense(128, activation='relu', name='hidden1')(inputs)
    x = Dense(32, activation='relu', name='hidden2')(x)
    x = Dense(32, activation='relu', name='hidden3')(x)
    x = Dense(16, activation='relu', name='hidden4')(x)
    x = Dense(16, activation='relu', name='hidden5')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(3, activation='softmax', name = 'output')(x)
    model = Model(inputs=inputs, output = prediction)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
def create_hyperparameters():
    batches = [2,4,6,8,10]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.3, 5)
    return {
        "model__batch_size":batches, 
        "model__optimizer":optimizers, 
        "model__keep_prob":dropout
    }

from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn= build_Model, verbose=1)
hyperparameters = create_hyperparameters()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])

# from sklearn.pipeline import make_pipeline
# pipe = make_pipeline(MinMaxScaler(), SVC(C=100)) # pipe = Pipeline([('minmaxscaler', MinMaxScaler()), ('kerasclassifier', model)])

from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(estimator=pipe,
                            param_distributions=hyperparameters,
                            n_iter=10, n_jobs=1, cv=3, verbose=1)
search.fit(x_train,y_train)

model = build_Model(search.best_params_["model__keep_prob"], search.best_params_['model__optimizer'])
model.fit(x_train,y_train, epochs=200, batch_size = search.best_params_["model__batch_size"])
loss, acc = model.evaluate(x_test, y_test)
# y_pridect = model.predict(x_test)
print("Best_params: ", search.best_params_)
print("Best_Acc   : ", acc)
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