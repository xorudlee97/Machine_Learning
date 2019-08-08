import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

file_dir = os.path.dirname("D:/LTK_AI_Study/Machine_Learning/Data/")
# 데이터 읽기
wine = pd.read_csv(file_dir+"/winequality-white.csv", sep=";", encoding='utf-8')
# 데이터 불리 하기
y = wine["quality"]
x = wine.drop("quality", axis= 1)

# y레이블 변경하기
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 학습하기
model = RandomForestClassifier(n_estimators=500, random_state=1, max_features=2, oob_score=False)
model.fit(x_train, y_train)
y_score = model.score(x_test, y_test)

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률=", accuracy_score(y_test, y_pred))
print(y_score)

'''
머신 러닝과 딥러닝의 모델링 구조는 차이가 없다.
머신 러닝
model = RandomFrest()
model.fit(x_train, y_train)
acc = model.scroe(x_test, y_test)
y_pred = model.predict(x_test)

딥러닝
model = Sequenecial()
model.fit(x_train, y_train)
loss, acc = model.eveludate(*)
'''