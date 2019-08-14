# 비지도 학습

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
x_Scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA
# 차원 축소(row 값 변경)
pca = PCA(n_components = 2)
pca.fit(x_Scaled)

x_pca = pca.transform(x_Scaled)
print("원본 데이터 형태 : ", x_Scaled.shape)
print("축소 데이터 형태 : ", x_pca.shape)