from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
import numpy as np
import tensorflow as tf

import os

scaler = StandardScaler()
# scaler = MinMaxScaler()

file_dir = os.path.dirname("D:/LTK_AI_Study/Machine_Learning/Data/")

# seed 값 생성
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = np.loadtxt(file_dir+"/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
scaler.fit(X)
X = scaler.transform(X)
Y = dataset[:,8]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8, shuffle=True)

# 모델의 구성
model = Sequential()
def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape = (8,), name= 'inputs')
    x = Dense(128, activation='relu', name='hidden1')(inputs)
    x = Dense(32, activation='relu', name='hidden2')(x)
    x = Dense(32, activation='relu', name='hidden3')(x)
    x = Dense(16, activation='relu', name='hidden4')(x)
    x = Dense(16, activation='relu', name='hidden5')(x)
    # x = Dropout(keep_prob)(x)
    prediction = Dense(1, activation='sigmoid', name = 'output')(x)
    model = Model(inputs=inputs, outputs = prediction)
    model.compile(optimizer=optimizer, loss= 'binary_crossentropy',metrics=['accuracy'])
    return model
def create_hyperparameters():
    batches = [2,4,6,8,10]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.3, 5)
    activation = ['relu','sigmoid']
    return{"model__batch_size":batches, "model__optimizer":optimizers, "model__keep_prob":dropout}

from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn= build_network, verbose=1)
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

model = build_network(search.best_params_["model__keep_prob"], search.best_params_['model__optimizer'])
model.fit(x_train,y_train, epochs=200, batch_size = search.best_params_["model__batch_size"])
loss, acc = model.evaluate(x_test, y_test)
# y_pridect = model.predict(x_test)
print("Best_params: ", search.best_params_)
print("Best_Acc   : ", acc)

'''
실습 1 케라스모델을 머신러닝 모델로 변경
실습 2 
'''