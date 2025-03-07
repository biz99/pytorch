import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

train_x = torch.FloatTensor([
    [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]    
])

train_y = torch.FloatTensor([
    [0.1, 1.5], [1, 2.8], [1.9, 4.1], [2.8, 5.4], [3.7, 6.7], [4.6, 8]
])

# 3.30 tensor의 훈련용 데이터 set 생성
train_dataset = TensorDataset(train_x, train_y)
# batch_size = 2로 선언하여 한 번의 배치 때마다 두 개의 샘플 데이터와 정답을 가져온다.
# shuffle = True의 참값 적용으로 불러오는 데이터의 순서를 무작위로 변경한다.
# drop_last로 배치 크기에 맞지 않는 배치를 제거한다. (ex. 전체 데이터 세트의 크기가 5일 때 배치 크기가 2라면 마지막 배치의 크기는 1이 된다. 따라서 크기가 1인 마지막 배치를 학습에 포함하지 않는 다는 뜻이다.)
train_dataloader = DataLoader(train_dataset, batch_size = 2, shuffle = True, drop_last = True)


model = nn.Linear(2, 2, bias = True)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)

for epoch in range(20000):
    cost = 0.0
    
    for batch in train_dataloader:
        x, y = batch
        output = model(x)
        
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        cost += loss
        
    cost = cost / len(train_dataloader)
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch + 1: 4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")