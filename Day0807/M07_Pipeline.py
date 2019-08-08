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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(MinMaxScaler(), SVC(C=100))

pipe.fit(x_train, y_train)

print("테스트 점수:", pipe.score(x_test, y_test))
