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
from keras.layers import Input, Dense
from keras.models import Model

encoding_dim = 32
input_img = Input(shape=(784,))
# encode 는 입력의 인코딩된 표현
encode_model = Dense(encoding_dim, activation='relu')(input_img)
# 레이어 추가
decode_model = Dense(784, activation = 'sigmoid')(encode_model)
# decode_model = Dense(784, activation = 'relu')(encode_model)

# 입력을 입력의 재구성으로 매피할 모델 784 -> 32 -> 784
autoencoder = Model(input_img, decode_model)

# 입력된 인코딩된 입력의 표현으로 매핑 784 -> 32
encoder = Model(input_img, encode_model)

# 인코딩된 입력을 위한 플레이스 홀더
encode_input = Input(shape=(encoding_dim,))
# 오토 인코더 모델의 마지막 레이어 얻기
decode_layer = autoencoder.layers[-1]
# 디 코더 모델 생성 32 -> 784
decoder = Model(encode_input, decode_layer(encode_input)) 

autoencoder.summary()
encoder.summary()
decoder.summary()

autoencoder.compile(optimizer = 'adadelta',
                    loss = 'binary_crossentropy', metrics=['accuracy'])
                    #loss = 'mse', metrics=['accuracy'])
history = autoencoder.fit(x_train, x_train,
                    epochs=50, batch_size=256,
                    shuffle=True, validation_data=(x_test, x_test))

# 숫자들을 인코딩 디코딩
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
print(loss, acc)