from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 기온 데이터 읽어 들이기
file_dir = os.path.dirname("D:/LTK_AI_Study/Machine_Learning/Data/")
df = pd.read_csv(file_dir+"/tem10y.csv", sep=",", encoding='utf-8')

# 데이터를 학습전용과 테스트 전용으로 분리
train_year = (df["연"] <= 2015)
test_year = (df["연"] >= 2016)
interval = 6

# 과거 6일의 데이터를 기반으로 학습할 데이터 만들기
def make_data(data):
    x = [] # 학습 데이터
    y = [] # 경과
    temps = list(data["기온"])
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return (x, y)

x_train, y_train = make_data(df[train_year])
x_test, y_test = make_data(df[test_year])

parameters = {
    "splitter":['best','random'],
    "max_features":['auto','sqrt','log2'],
    "max_depth":[3,6,9,12,15],
    "random_state":[0,1,2,3,4,5]
}

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold

lr = DecisionTreeRegressor()
kflod_cv = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(estimator=lr, param_distributions=parameters, cv=kflod_cv, n_iter=10, n_jobs=-1, verbose=1)
model.fit(x_train,y_train)
print("훈련 점수  :",model.score(x_train, y_train))
print("테스트 점수:",model.score(x_test, y_test))

