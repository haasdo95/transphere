import torch
from torch import nn
from layers import *
import torch.nn.functional as F


class Impaint(nn.Module):
    def __init__(self):
        super(Impaint, self).__init__()
        self.conv1 = ChebConv(1, 16, skip=False, kernel_size=3)
        self.conv2 = ChebConv(16, 64, skip=False, kernel_size=3)
        self.conv3 = ChebConv(64, 16, skip=False, kernel_size=3)
        self.fc = ChebConv(16, 1, skip=False, kernel_size=1)  # effectively fully-conn

    def forward(self, laplacian, inputs):
        out = F.relu(self.conv1(laplacian, inputs))
        out = F.relu(self.conv2(laplacian, out))
        out = F.relu(self.conv3(laplacian, out))
        out = self.fc(laplacian, out)
        return out

