"""从原 main.py 提取的网络定义，无训练副作用。"""
import torch


class TitanicModel(torch.nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, 16), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(16, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))

    def forward(self, x):
        return self.net(x)
