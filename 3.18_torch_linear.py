import torch
# torch.optim은 최적화 함수가 포함되어 있는 모듈이다.
from torch import optim
# 예제 30개
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

# reqires_grad는 텐서에 대한 연산을 추적하여 역전파 메서드를 호출해 기울기를 생성하고 저장한다.
# 파이토치에서 지원하는 자동미분 기능의 사용여부라고 볼 수 있다.
weight = torch.zeros(1, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)
learning_rate = 0.001

# 최적화 선언
optimizer = optim.SGD([weight, bias], lr = learning_rate)

# optimizer = torch.optim.SGD(
#     params, #변수
#     lr, #학습률
#     **kwargs
# )

# 3.22 에폭, 가설, 손실 함수 선언
# 확률적 경사하강법 사용
for epoch in range(10000):
    hypothesis = x * weight + bias
    cost = torch.mean((hypothesis - y)**2)
    
    # 3.25 직접 가중치에대한 기울기와 값 확인
    print(f"Epoch : {epoch+1:4d}")
    print(f"Step [1] : Gradient : {weight.grad}, Weight : {weight.item():.5f}")
    
    # 3.23 가중치와 편향 갱신(파이토치)
    optimizer.zero_grad()
    print(f"Step [2] : Gradient : {weight.grad}, Weight : {weight.item():.5f}")
    
    cost.backward()
    print(f"Step [3] : Gradient : {weight.grad}, Weight : {weight.item():.5f}")
    
    optimizer.step()
    print(f"Step [4] : Gradient : {weight.grad}, Weight : {weight.item():.5f}")

    # if (epoch + 1) % 1000 == 0:
    #     print(f"Epoch : {epoch+1:4d}, Weight : {weight.item():.3f}, Bias : {bias.item():.3f}, Cost : {cost:.3f}")
        
    
        