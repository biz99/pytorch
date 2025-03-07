import torch
import numpy as np

ndarray = np.array([1,2,3], dtype = np.uint8)
# np.uint8 numpy 데이터 타입

print(torch.tensor(ndarray))
print(torch.Tensor(ndarray))
print(torch.from_numpy(ndarray))
# numpy와 매우 친화적인 구조를 가지고 있다.