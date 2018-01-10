import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 32, 5, bias=False)  # 40 -> 18
        nn.init.kaiming_normal(self.conv1.weight.data)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)  # 16 -> 8
        nn.init.kaiming_normal(self.conv2.weight.data)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, bias=False)  # 8 -> 6
        nn.init.kaiming_normal(self.conv3.weight.data)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, bias=False, padding=1)  # 6 -> 6
        nn.init.kaiming_normal(self.conv4.weight.data)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 3, bias=False)  # 6 -> 4
        nn.init.kaiming_normal(self.conv5.weight.data)
        self.bn6 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 ** 2, 256)
        nn.init.kaiming_normal(self.fc1.weight.data)
        self.bn7 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)
        nn.init.kaiming_normal(self.fc2.weight.data)

    def forward(self, x):
        x = self.bn1(x)
        x = self.bn2(F.relu(F.max_pool2d(self.conv1(x), 2)))
        x = self.bn3(F.relu(F.max_pool2d(self.conv2(x), 2)))
        x = self.bn4(F.relu(self.conv3(x)))
        x = self.bn5(F.relu(self.conv4(x)))
        x = self.bn6(F.relu(self.conv5(x)))
        x = x.view(x.size(0), 128 * 4 ** 2)
        x = self.bn7(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, 1)
