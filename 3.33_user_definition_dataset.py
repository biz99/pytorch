import pandas as pd
import torch
from torch import nn


class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.x = df.iloc[:,0].values
        self.y = df.iloc[:,1].values
        self.length = len(df)
        
    def __getitem__(self, index):
        x = torch.FloatTensor([self.x[index]**2, self.x[index]])
        y = torch.FloatTensor([self.y[index]])
        
        return x, y
    
    def __len__(self):
        return self.length
    

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2,1)
    
    def forward(self, x):
        x = self.layer(x)
        return x