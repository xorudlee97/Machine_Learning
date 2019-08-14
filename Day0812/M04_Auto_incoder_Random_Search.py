from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_train.shape[1:])))
print(x_train.shape)        # (60,000, 784)
print(x_test.shape)         # (10,000, 784)

##############모델 구성############
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras import regularizers


def build_Model(keep_prob=0.5, optimizer='adam', kernel_regularizer_num=0.0001):
    # encode 는 입력의 인코딩된 표현
    encoding_dim = 32
    input_img = Input(shape=(784,))
    encode_model = Dense(encoding_dim, activation='relu', kernel_regularizer= regularizers.l1(kernel_regularizer_num))(input_img)
    input_layer = BatchNormalization()(encode_model)
    # input_layer = Dropout(0.25)(input_layer)
    input_layer = Dense(encoding_dim, activation='relu', kernel_regularizer= regularizers.l2(kernel_regularizer_num))(input_layer)
    # 레이어 추가
    input_layer = Dense(128, activation = 'relu')(input_layer)
    input_layer = Dense(128, activation = 'relu')(input_layer)
    input_layer = Dense(128, activation = 'relu')(input_layer)
    input_layer = Dropout(keep_prob)(input_layer)

    input_layer = Dense(encoding_dim, activation = 'relu')(input_layer)
    # decode는 입력의 손실있는 재구성
    decode_model = Dense(784, activation = 'sigmoid')(input_layer)

    # 입력을 입력의 재구성으로 매피할 모델 784 -> 32 -> 784
    autoencoder = Model(input_img, decode_model)
    # 디 코더 모델 생성 32 -> 784
    autoencoder.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
    return autoencoder
def create_hyperparameters():
    batches = [32,64,128,256,512]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.3, 5)
    kernel_regularizers = [0.1, 0.001, 0.0001, 0.00001]
    return{
        "model__batch_size":batches, 
        "model__optimizer":optimizers, 
        "model__keep_prob":dropout,
        "model__kernel_regularizer_num":kernel_regularizers
    }
from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn= build_Model, verbose=1)
hyperparameters = create_hyperparameters()

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

pipe = Pipeline([('scaler', MinMaxScaler()), ('model', model)])

from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(estimator=pipe,
                            param_distributions=hyperparameters,
                            n_iter=10, n_jobs=-1, cv=3, verbose=1)
search.fit(x_train,x_train)

autoencoder = build_Model(search.best_params_["model__keep_prob"], search.best_params_['model__optimizer'], search.best_params_['model__kernel_regularizer_num'])
print(search.best_params_)
history = autoencoder.fit(x_train, x_train,
                    epochs=100, batch_size=search.best_params_["model__batch_size"],
                    shuffle=True, validation_data=(x_test, x_test))

# 숫자들을 인코딩 디코딩
encoding_dim = 32
input_img = Input(shape=(784,))
encode_model = Dense(encoding_dim, activation='relu', kernel_regularizer= regularizers.l1(search.best_params_['model__kernel_regularizer_num']))(input_img)
encode_input = Input(shape=(encoding_dim,))
decode_layer = autoencoder.layers[-1]
encoder = Model(input_img, encode_model)
decoder = Model(encode_input, decode_layer(encode_input)) 

# 인코딩된 입력을 위한 플레이스 홀더
encode_img = encoder.predict(x_test)
decode_img = decoder.predict(encode_img)

print(encode_img)
print(decode_img)
print(encode_img.shape)
print(decode_img.shape)

##########################이미지 출력####################
import matplotlib.pyplot  as plt

# 숫자의 개수
number = 10
plt.figure(figsize=(20, 4))
for i in range(number):
    #원본 데이터
    ax = plt.subplot(2, number, i + 1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 재지정 데이터
    ax = plt.subplot(2, number, i + 1 + number)
    plt.imshow(decode_img[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

################## 그래프 그리기#################
def plot_acc(history, title=None):
    if not isinstance(history, dict):
        history = history.history
    
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Accracy')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc=0)

def plot_loss(history, title=None):
    if not isinstance(history, dict):
        history = history.history
    
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc=0)

plot_acc(history, '학습 경과에 따른 정확도 변화 추이')
plt.show()
plot_loss(history, '학습 경과에 따른 손실된 변화 추이')
plt.show()

loss, acc = autoencoder.evaluate(x_test, x_test)
print(search.best_params_)
print(loss, acc)