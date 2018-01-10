import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7, bias=False)  # 40 -> 34
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 17
        self.conv2 = nn.Conv2d(8, 10, kernel_size=5, bias=False)  # 13
        self.bn2 = nn.BatchNorm2d(10)
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 6
        self.conv3 = nn.Conv2d(10, 10, kernel_size=3, bias=False)  # 4
        self.bn3 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(10 * 4 * 4, 32, bias=False)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 3 * 2)
        self.fc2.weight.data.fill_(0)
        self.fc2.bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, x):
        theta = self.bn1(F.relu(self.pool1(self.conv1(x))))
        theta = self.bn2(F.relu(self.pool2(self.conv2(theta))))
        theta = self.bn3(F.relu(self.conv3(theta)))
        theta = self.bn4(F.relu(self.fc1(theta.view(theta.size(0), -1))))
        theta = self.fc2(theta).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x
