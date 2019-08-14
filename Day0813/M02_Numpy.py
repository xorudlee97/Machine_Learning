# # # 모델 저장
# # model.save("savbetest.h5")
# # # 모델 블러 오기
# # from keras.models import load_model
# # model = load_model("savetest.ht")
# # from keras.layers import Dense
# # model.add(Dense(1))

# ######################################
# # import numpy as np
# # a = np.arange(10)

# # # Numpy 저장
# # print(a)
# # np.save("aaa.npy", a)
# # b = np.load("aaa.npy")
# # priunt(b)
# #####################################

# # import numpy
# # import pandas as pd
# # dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter=',')
# # iris_data = pd.read_csv("./data/iris.csv", encoding='utf-8')
# #                         # index_col = 0, encoding='cp949', seq = ',', header = None
# #                         # names = ['x1', 'x2', 'x3', 'x4', 'y']
# # wine = pd.read_csv("./data/winequality-white.csv", seq=',', encoding='utf-8')

# ######### utf-8 #######
# #-*- coding: utf-8-*-
# ######### 한글처리######

# ##### 각종 샘플 데이터 셋 ###
# import numpy as np
# import pandas as pd

# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# np.save("mnist_data.npy", [x_train, y_train, x_test, y_test])

# from keras.datasets import cifar10
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# np.save("cifar10_data.npy", [x_train, y_train, x_test, y_test])

# from keras.datasets import boston_housing
# (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# np.save("boston_data.npy", [x_train, y_train, x_test, y_test])

# from sklearn.datasets import load_boston
# boston = load_boston()
# np.save("boston_housing_train_data.npy", boston.data)
# np.save("boston_housing_test_data.npy", boston.target)

# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
# np.save("cancer_train_data.npy", cancer.data)
# np.save("cancer_test_data.npy", cancer.target)

# import os

# file_dir = os.path.dirname("D:/LTK_AI_Study/Machine_Learning/Data/")

# dataset = np.loadtxt(file_dir+"/pima-indians-diabetes.csv", delimiter=',')
# iris_data = pd.read_csv(file_dir+"/iris.csv", encoding='utf-8')
#                         # index_col = 0, encoding='cp949', sep = ',', header = None
#                         # names = ['x1', 'x2', 'x3', 'x4', 'y']
# wine = pd.read_csv(file_dir+"/winequality-white.csv", sep=';', encoding='utf-8')
# np.save("pima.npy", dataset)
# np.save("iris_data.npy", iris_data)
# np.save("wine.npy", wine)

# pima = np.load("pima.npy")
# iris_data = np.load("iris_data.npy")
# wine = np.load("wine.npy")
# mnist_data = np.load("mnist_data.npy")
# cifar10_data = np.load("cifar10_data.npy")
# boston_data = np.load("boston_data.npy")
# boston_housing_train_data = np.load("boston_housing_train_data.npy")
# boston_housing_test_data = np.load("boston_housing_test_data.npy")
# cancer_train_data = np.load("cancer_train_data.npy")
# cancer_test_data = np.load("cancer_test_data.npy")

# print(pima.shape)
# print(iris_data.shape)
# print(mnist_data.shape)
# print(boston_data.shape)
# print(boston_housing_train_data.shape)
# print(boston_housing_test_data.shape)
# print(cancer_train_data.shape)
# print(cancer_test_data.shape)
# print(wine.shape)

    
import numpy as np
import os
file_dir = os.path.dirname("D:/LTK_AI_Study/Machine_Learning/Data/")
save_dir = os.path.dirname("D:/LTK_AI_Study/Machine_Learning/Numpy/")
a = np.arange(10)
print(a)
np.save(save_dir+"aaa.npy",a)
b = np.load(save_dir+"aaa.npy")
print(b)

# ######모델저장####
# model.save("savetest01.h5")
# #####모델불러오기#####
# from keras.models import load_model
# model = load_model("savetest01.h5")
# from keras.layers import Dense
# model.add(Dense(1))

########pandas to numpy#############3
# pandas.value

########## csv 불러오기############
# dataset = numpy.loadtxt("000.csv",delimiter=",")
# iris_data = pd.read_csv("000.csv",encoding="utf-8")
        # index_col = 0, encoding="cp949", sep=",", header=none
        #names=["x1","x2"]
# wine = pd.read_csv("000.csv", sep=",",encoding="utf-8")

####utf-8#####
# -*-coding: utf-8 -*-
#####sample######
def name_class(y):
    for i in range(len(y)):
        if y[i] == b"Iris-setosa":
            y[i] = 0
        elif y[i] == b"Iris-versicolor":
            y[i] = 1
        else:
            y[i] = 2

    return y
import pandas as pd
from keras.datasets import mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()



mnist_x = np.vstack((np.array(x_train), np.array(x_test)))
mnist_y = np.hstack((np.array(y_train), np.array(y_test)))
np.save(save_dir+"/mnist_x.npy",mnist_x)
np.save(save_dir+"/mnist_y.npy",mnist_y)
x = np.load(save_dir+"/mnist_x.npy")
y = np.load(save_dir+"/mnist_y.npy")

# mnist_train =
print(x.shape)
print(y.shape)




from keras.datasets import cifar10
(x_train, y_train),(x_test,y_test) = cifar10.load_data()
cifar10_x = np.vstack((np.array(x_train), np.array(x_test)))
cifar10_y = np.vstack((np.array(y_train), np.array(y_test)))
np.save(save_dir+"/cifar10_x.npy",cifar10_x)
np.save(save_dir+"/cifar10_y.npy",cifar10_y)
x = np.load(save_dir+"/cifar10_x.npy")
y = np.load(save_dir+"/cifar10_y.npy")
# mnist_train =
print(x.shape)
print(y.shape)

from keras.datasets import boston_housing
(x_train, y_train),(x_test,y_test) = boston_housing.load_data()
boston_housing_x = np.vstack((np.array(x_train), np.array(x_test)))
boston_housing_y = np.hstack((np.array(y_train), np.array(y_test)))

np.save(save_dir+"/boston_housing_x.npy",boston_housing_x)
np.save(save_dir+"/boston_housing_y.npy",boston_housing_y)
x = np.load(save_dir+"/boston_housing_x.npy")
y = np.load(save_dir+"/boston_housing_y.npy")
# mnist_train =
print("boston_housingx",x.shape)
print("boston_housingy",y.shape)




# from skleran.datasets import load_boston

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
label = cancer.target.reshape(-1,1)
print("cancer_ori",cancer.data.shape)
print("cancer_ori",cancer.target.shape)
cancer_data = np.c_[cancer.data,label]

np.save(save_dir+"/cancer_data.npy",cancer_data)
cancer_d = np.load(save_dir+"/cancer_data.npy")
print("cancer",cancer_d.shape)





iris_data = pd.read_csv(file_dir+"/iris2.csv", encoding="utf-8")


x = np.array(iris_data.iloc[:,:-1])
y = name_class(iris_data.iloc[:,-1])

y = np.array(y,dtype=np.int32)
iris2_data = np.c_[x,y]
np.save(save_dir+"/iris2_data.npy",iris2_data)
# np.save("iris2_label.npy",y)

iris2_data = np.load(save_dir+"/iris2_data.npy")
# iris2_label = np.load("./iris2_label.npy")

print("iris2_data:",iris2_data.shape)
# print("iris2_label:",iris2_label.shape)

wine_data = pd.read_csv(file_dir+"/winequality-white.csv",sep=";", encoding="utf-8")
# print(wine_data)
np.save("wine_data.npy",np.array(wine_data))
wine = np.load("wine_data.npy")
print("whine:",wine.shape)
pima = pd.read_csv(file_dir+"/pima-indians-diabetes.csv",header = None)
# print(pima)
np.save(save_dir+"/pima.npy",np.array(pima))
pima = np.load(save_dir+"/pima.npy")
print("pima:",pima.shape)

