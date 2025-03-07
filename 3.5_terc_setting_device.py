import torch

device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built else "cpu"

cpu = torch.FloatTensor([1,2,3])
gpu = cpu.to(device)
tensor = torch.rand((1,1), device=device)

print(device)
print(cpu)
print(gpu)
print(tensor)

# 장치변환
# gpu2 = cpu.to("mps")
# gpu2 = cpu.to(device)

#print(gpu2)