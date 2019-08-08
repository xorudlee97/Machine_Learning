import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

file_dir = os.path.dirname("D:/LTK_AI_Study/Machine_Learning/Data/")
iris_data = pd.read_csv(file_dir+"/iris.csv", encoding='utf-8', names=['SepalLength','SepalWidth','PataLength','PatalWidth','Name'])
# print(iris_data)
# print(iris_data.shape)
# print(type(iris_data))

# 레이블로 자르기
y = iris_data.loc[:, 'Name']
x = iris_data.loc[:, ['SepalLength', 'SepalWidth', 'PataLength', 'PatalWidth']]

# 열로 자르기
# x = iris_data.iloc[:, 4]
# y = iris_data.iloc[:,0:4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size = 0.8, shuffle=True)

# 학습 하기
# clf = SVC()
# clf = LinearSVC()
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(x_train, y_train)

# 평가하기
y_pred = clf.predict(x_test)
print("정답률:", accuracy_score(y_test, y_pred))
