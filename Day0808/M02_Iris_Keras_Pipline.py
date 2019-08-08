import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np
import os

file_dir = os.path.dirname("D:/LTK_AI_Study/Machine_Learning/Data/")
iris_data = pd.read_csv(file_dir+"/iris.csv", encoding='utf-8', names=['SepalLength','SepalWidth','PataLength','PatalWidth','Name'])

# 레이블로 자르기
y = iris_data.loc[:, 'Name']
x = iris_data.loc[:, ['SepalLength', 'SepalWidth', 'PataLength', 'PatalWidth']]

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


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input

model = Sequential()
def build_Model(keep_prob=0.3, optimizer='adam'):
    inputs = Input(shape=(4,), name='inputs')
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
print("Best_params: ", search.best_params_)
print("Best_Acc   : ", acc)

y_pridect = model.predict(x_test)
y_pridect = np.round(y_pridect).astype(int)
y_pridect = np.argmax(y_pridect, axis=1)
y_pridect = lable.inverse_transform(y_pridect)
print(y_pridect)