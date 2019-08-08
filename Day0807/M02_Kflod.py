import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import os

import warnings
warnings.filterwarnings('ignore')


# 기온 데이터 읽어 들이기
file_dir = os.path.dirname("D:/LTK_AI_Study/Machine_Learning/Data/")
iris_data = pd.read_csv(file_dir+"/iris2.csv", encoding='utf-8')

# 붓꽃 데이터를 분리
y = iris_data.loc[:,"Name"]
x = iris_data.loc[:,["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

# classfier 알고리즘 모두 추출하기
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")
# allAlgorithms = all_estimators(type_filter="regressor")

# K 분할 크로스 벨리데이션 전용 객체
kflod_cv  = KFold(n_splits=5, shuffle=True)
for (name, algorithms) in allAlgorithms:
    clf = algorithms()
    
    if hasattr(clf, "score"):
        # 크로스 벨리데이션
        scores = cross_val_score(clf, x, y, cv=kflod_cv)
        print(name,"의 정답률")
        print(scores)