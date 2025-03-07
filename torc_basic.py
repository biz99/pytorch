import torch

# # MPS 디바이스 설정
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# # Tensor를 MPS(GPU)로 이동
# x = torch.randn(3, 3).to(device)
# print(x)


print(torch.__version__)
print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())