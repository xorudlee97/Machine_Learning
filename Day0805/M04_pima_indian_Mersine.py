from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy
import tensorflow as tf

import os

scaler = MinMaxScaler()

file_dir = os.path.dirname("D:/LTK_AI_Study/Machine_Learning/Data/")

# seed 값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = numpy.loadtxt(file_dir+"/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
scaler.fit(X)
X = scaler.transform(X)
Y = dataset[:,8]

#모델의 실행
# model = SVC()
# model = LinearSVC()
model = KNeighborsClassifier(n_neighbors=1)

# 모델 실행
model.fit(X,Y)
y_predict = model.predict(X)
# 결과 출력
print("\n Acc: %.4f"%(accuracy_score(Y, y_predict)))

'''
실습 1 케라스모델을 머신러닝 모델로 변경
실습 2 
'''