from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import pandas as pd
import os

import warnings
warnings.filterwarnings('ignore')

# 기온 데이터 읽어 들이기
file_dir = os.path.dirname("D:/LTK_AI_Study/Machine_Learning/Data/")
iris_data = pd.read_csv(file_dir+"/iris2.csv", encoding='utf-8')

y = iris_data.loc[:,"Name"]
x = iris_data.loc[:,["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

# 학습 전용과 테스트 전용 분리
warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

# 그리드 서치에서 사용할 매게 변수
parameters = {"C":[1,10,100,1000], "kernel":["linear","rbf","sigmoid"], "gamma":[0.001,0.0001]}

# 그리드 서치
kflod_cv = KFold(n_splits=5, shuffle=True)
clf = RandomizedSearchCV(estimator=SVC(), param_distributions=parameters, n_iter=10, n_jobs=1, cv=3, verbose=1)
clf.fit(x_train, y_train)
print("최적의 매개 변수=", clf.best_estimator_)

# 최적의 매개변수로 평가
y_pred = clf.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print("죄종 정답률 = ", last_score)