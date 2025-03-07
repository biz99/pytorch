import torch

tensor = torch.rand(1,2)
print(tensor)
print(tensor.shape)

# 차원 변환
tensor = tensor.reshape(2,1)
print(tensor)
print(tensor.shape)