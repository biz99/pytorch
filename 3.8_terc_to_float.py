import torch

# 텐서를 numpy 배열로 변환
tensor = torch.FloatTensor([1,2,3])
# detach 명령어를 통해 연산 그래프(Graph)에서 분리된 새로운 텐서를 반환한다.
ndarray = tensor.detach().cpu().numpy()
print(ndarray)
print(type(ndarray))

# 텐서는 기존 데이터 형식과 다르게 학습을 위한 데이터 형식으로 모든 연삭을 추적해 기록한다. 이 기록을 통해 역전파 (Backpropagation) 등 과 같은 연산이 진행돼 모델 학습이 이루어진다.
# 텐서는 모델 학습에 특화된 데이터 이므로, 정확한 데이터 형식을 취해 활용해야한다.
