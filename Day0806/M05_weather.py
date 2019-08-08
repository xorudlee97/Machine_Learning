from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinmaxScaler, StandardScaler
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

# 직선 회귀 분석
lr = LinearRegression(normalize= True)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

aaa = lr.score(x_test, y_test)
print(aaa)

# 결과 그래프 그리기
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(y_test, c="r")
plt.plot(y_pred, c="b")
plt.savefig('tenki-kion-lr.png')
plt.show()