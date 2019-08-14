from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers

# CIFAR_10은 3채널로 구성된 32*32 이미지 60000장을 갖는다.
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

# 데이터 셋 불러 오기
(x_train, _), (x_test, _) = cifar10.load_data()
x_train_Row = x_train.shape[0]
x_test_Row = x_test.shape[0]

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = x_train.reshape((len(x_train), IMG_ROWS, IMG_COLS, IMG_CHANNELS))
x_train =x_train[:10]
x_test = x_test.reshape((len(x_test), IMG_ROWS, IMG_COLS, IMG_CHANNELS))
print(x_train.shape)
print(x_test.shape)


from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras import regularizers

encoding_dim = 32
input_img = Input(shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS))
# encode 는 입력의 인코딩된 표현
encode_model = Dense(encoding_dim, activation='relu')(input_img)
# 레이어 추가
input_layer = Dense(128, activation = 'relu')(encode_model)
input_layer = Dense(128, activation = 'relu')(input_layer)
input_layer = Dense(128, activation = 'relu')(input_layer)
input_layer = Dense(encoding_dim, activation = 'relu')(input_layer)
decode_model = Dense(3, activation = 'sigmoid')(input_layer)
autoencoder = Model(input_img, decode_model)

autoencoder.compile(optimizer = 'RMSprop',
                    # loss = 'categorical_crossentropy', metrics=['accuracy'])
                    loss = 'binary_crossentropy', metrics=['accuracy'])
                    #loss = 'mse', metrics=['accuracy'])
history = autoencoder.fit(x_train, x_train,
                    epochs=50, batch_size=5,
                    shuffle=True, validation_data=(x_test, x_test))
# 숫자들을 인코딩 디코딩
decode_img = autoencoder.predict(x_test)

print(decode_img)
print(decode_img.shape)

##########################이미지 출력####################
import matplotlib.pyplot  as plt

# 숫자의 개수
number = 10
plt.figure(figsize=(20, 4))
for i in range(number):
    #원본 데이터
    ax = plt.subplot(2, number, i + 1)
    plt.imshow(x_test[i].reshape(IMG_ROWS,IMG_COLS, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 재지정 데이터
    ax = plt.subplot(2, number, i + 1 + number)
    plt.imshow(decode_img[i].reshape(IMG_ROWS,IMG_COLS, 3))
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
print(acc)