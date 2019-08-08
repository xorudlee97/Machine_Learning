import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
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

# classfier 알고리즘 모두 추출하기
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")
# allAlgorithms = all_estimators(type_filter="regressor")

# print(allAlgorithms)
# print(len(allAlgorithms))
# print(type(allAlgorithms))

for(name, algorithm) in allAlgorithms:
    # 알고리즘 객체 생성
    clf = algorithm()

    # 학습 하고 평가하기
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(name, "의 정답률 = ", accuracy_score(y_test, y_pred))
