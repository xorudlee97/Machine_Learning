from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from keras.utils import np_utils
import numpy as np
cancer = load_breast_cancer() # 분류

x = cancer.data
y = cancer.target
x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, train_size = 0.8, shuffle=True
)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = Sequential()
def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape = (30,), name= 'inputs')
    x = Dense(128, activation='relu', name='hidden1')(inputs)
    x = Dense(32, activation='relu', name='hidden2')(x)
    x = Dense(32, activation='relu', name='hidden3')(x)
    x = Dense(32, activation='relu', name='hidden4')(x)
    x = Dense(16, activation='relu', name='hidden5')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(1, activation='sigmoid', name = 'output')(x)
    model = Model(inputs=inputs, outputs = prediction)
    model.compile(optimizer=optimizer, loss= 'binary_crossentropy',metrics=['accuracy'])
    return model
def create_hyperparameters():
    batches = [3,6,9,12,15,18,21,27,30]
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
# print(search.best_params_)
model = build_network(search.best_params_["keep_prob"], search.best_params_['optimizer'])
model.fit(x_train,y_train, epochs=200, batch_size = search.best_params_["batch_size"])
loss, acc = model.evaluate(x_test, y_test)
y_pridect = model.predict(x_test)
print(search.best_params_)
print(acc)
# print(y_pridect)