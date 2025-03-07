import torch

# nn 은 신경망 패키지가 포함되어 있다.
from torch import nn
# 최적화 함수 모듈
from torch import optim

x = torch.FloatTensor(
    [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],
     [11],[12],[13],[14],[15],[16],[17],[18],[19],[20],
     [21],[22],[23],[24],[25],[26],[27],[28],[29],[30]
     ]
)

y = torch.FloatTensor(
    [
     [0.94],[1.98],[2.88],[3.92], [3.96],[4.55],[5.64],[6.3],[7.44],[9.1],
     [8.46],[9.5],[10.67],[11.16],[14],[11.83],[14.4],[14.25],[16.2],[16.32],
     [17.46],[19.8],[18],[21.34],[22],[22.5],[24.57],[26.04],[21.6],[28.8]
    ]
)

model = nn.Linear(1, 1, bias = True)
# 평균 제곰 오차 클래스
criterion = torch.nn.MSELoss()
learning_rate = 0.001
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 3.27 에폭, 가설, 손실 함수 선언을 대체하는 코드
for epoch in range(10000):
    output = model(x)
    cost = criterion(output, y)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        # model.parameters() : weight, bias 변수를 사용하지 않으므로, 학습결과 출력 시 모델 매게변수(model.parameters())를 출력한다.
        print(f"Epoch : {epoch + 1: 4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")