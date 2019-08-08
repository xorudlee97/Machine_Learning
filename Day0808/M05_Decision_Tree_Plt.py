from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x_train, x_test, y_trian, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42
)

tree = DecisionTreeClassifier(random_state=0, max_depth = 5, min_samples_split = 4, min_samples_leaf = 2, class_weight  ='balanced')
tree.fit(x_train, y_trian)
print("훈련 세트 정확도   : {:.3f}".format(tree.score(x_train, y_trian)))
print("테스트 세트 정확도 : {:.3f}".format(tree.score(x_test, y_test)))
print("특성의 중요도:\n", tree.feature_importances_)
# 단점
# 주어진 데이터 안쪽에서만 동작한다.

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.ylabel("특성 중요도") 
    plt.ylabel("특성")
    plt.ylim(-1, n_features)
plot_feature_importances_cancer(tree)
plt.show() 