import matplotlib.pyplot as plt
import pandas as pd
import os

# 데이터 읽기
file_dir = os.path.dirname("D:/LTK_AI_Study/Machine_Learning/Data/")
wine = pd.read_csv(file_dir+"/winequality-white.csv", sep=";", encoding='utf-8')

count_data = wine.groupby('quality')['quality'].count()
print(count_data)

# 수를 그래프로 그리기
count_data.plot()
plt.savefig('wine-count-plt.png')
plt.show()