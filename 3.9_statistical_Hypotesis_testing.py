# 1 가설 검정의 기본 개념

# 1. 귀무가설 (Null Hypothesis H0)
# "차이가 없다" 또는 "영향이 없다"는 가설
# ex) 이 약을 복용한 그룹과 복용하지 않은 그룹의 평균 혈압 차이는 없다.

# 2. 대립가설 (Alternative Hypothesis H1)
# "차이가 있다"는 가설, 귀무가설과는 반대되는 가설
# ex) 이 약을 복용한 그룹과 복용하지 않은 그룹의 평균 혈압 차이가 있다.

# 3. 유의수준 (Significance Level, a(알파))
# 일반적으로 0.05(%5)를 사용하며, 귀무가설이 참인데도 기각할 확률을 의미한다.

# 4. p-value (유의 확률)
# 귀무가설이 맞다고 가정했을 때, 현재의 데이터를 얻을 확률 
# ex)
# p-value < 0.05 : 귀무가설 기각 (H1 채택)
# p-value >= 0.05 : 귀무가설 채택 (유의미한 차이가 없음)

# 2 t-검정(t-test)
# t-검정은 두 개의 그룹 간 평균 차이가 통계적으로 유의미한지 판단하는 검정 방법
# 1. Paired t-test(대응표본 t-검정)
# - 같은 그룹에서 측정된 데이터를 비교할 때 사용
#  ex) 치료 전후, 학생들의 시험성적(중간, 기말), 같은 사람의 다이어트 전후 비교
# - d = X1 - X2
# 공식 : t = d / (Sd/n^1/2)
# d = 차이값의 평균
# Sd = 차이값의 표준편차
# n = 샘플의 개수
# 2. Unpaired t-test(독립표본 t-검정)
# - 서로 다른 두 그룹의 평균 차이를 비교할 때 사용
#  ex ) 남성과 여성의 키 비교, A반과 B반의 평균 성적 비교
# 공식 : t = (X1 - X2) / (s1^2/n1 + s2^2/n2)^(1/2)
# X1, X2 : 각 그룹의 평균
# s1^2, s2^2 : 각 그룹의 분산
# n1, n2 : 각 그룹의 샘플 개수

# 성별에 따른 키 검정 차이

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt

# stat.norm.rvs 는 특정 평균(loc), 표준편차(scale)를 분포에서 데이터를 샘플링하는 함수로 size = 500 을 통해 남녀 각각 500개의 데이터를 샘플링 했다.
man_height = stats.norm.rvs(loc = 170, scale =10, size = 500, random_state = 1)
woman_height = stats.norm.rvs(loc=150, scale = 10, size = 500, random_state = 1)

X = np.concatenate([man_height, woman_height])
Y = ["man"] * len(man_height) + ["woman"] * len(woman_height)

df = pd.DataFrame(list(zip(X, Y)), columns = ["X", "Y"])
fig = sns.displot(data = df, x = "X", hue = "Y", kind = "kde")
fig.set_axis_labels("cm", "count")
plt.show()

statistic, pvalue = stats.ttest_ind(man_height, woman_height, equal_var = True)
print("statistic", statistic)
print("pvalue : ", pvalue)
print("*", pvalue < 0.05)
print("**", pvalue < 0.001)

# 결과는 아래와 같이 나오므로, pvalue 가 0.05 보다 작으면 *를 표시하고, 0.001보다 작으면 **로 표시하도록 하여, 더 많이 유의하다고 판단할 수 있다. 따라서
# 0.001보다 작기 대문에 키는 성별을 구분하늗 데 매우 유의미한 변수라는 것을 확인할 수 있다.

# statistic 31.96162891312776
# pvalue :  6.22858543819892e-155
# * True
# ** True