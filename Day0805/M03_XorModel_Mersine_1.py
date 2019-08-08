from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

# 데이터 
# XOR 모델
#   0 1
# 0 0 1
# 1 1 0
# 한개의 그래프로는 0과 1을 구분 지을 수가 없다.

x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

# 모델
# model = LinearSVC()
model = SVC()

# 실행
model.fit(x_data, y_data)

# 예측

x_test = [[0,0],[1,0],[0,1],[1,1]]
y_predict = model.predict(x_test)

print(x_test, "의 예측 결과", y_predict)
print("acc = ", accuracy_score([0,1,1,0], y_predict))