# 단순선형회귀를 이용한 주어진 데이터셋을 학습하는 코드

import numpy as np

# 예제 30개
x = np.array(
    [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],
     [11],[12],[13],[14],[15],[16],[17],[18],[19],[20],
     [21],[22],[23],[24],[25],[26],[27],[28],[29],[30]
     ]
)

y = np.array(
    [
     [0.94],[1.98],[2.88],[3.92], [3.96],[4.55],[5.64],[6.3],[7.44],[9.1],
     [8.46],[9.5],[10.67],[11.16],[14],[11.83],[14.4],[14.25],[16.2],[16.32],
     [17.46],[19.8],[18],[21.34],[22],[22.5],[24.57],[26.04],[21.6],[28.8]
    ]
)

# 하이퍼파라미터 초기화 넘파이
weight = 0.0 #가중치
bias = 0.0 #편향
learning_rate = 0.005 #학습률

# 3.13 에폭 설정
# 에폭은 Forward Porpagation, Back Propagation 등의 모델연산을 전체 데이터 세트가 1회 통과하는 것을 의미한다.
for epoch in range(10000):

    # 에폭이 작을 경우 underfiiting, 에폭이 너무 많을 경우 overfitting이 발생할 수 있다.
    y_hat = weight * x + bias
    # mean 함수를 통해 산술평균을 계산한다.
    # MSE 활용
    cost = ((y-y_hat)**2).mean()

    # 3.15 가중치와 편향 갱신(넘파이)
    # 경사하강법 사용
    weight = weight - learning_rate *((y_hat-y)*x).mean()
    bias = bias - learning_rate * (y_hat -y).mean()

    #3.16 학습 기록 출력
    if(epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch+1:4d} Weight : {weight:.3f}, Bias : {bias:.3f}, Cost : {cost:.3f}")

