import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.attention import AttentionModule

__author__ = "prlz77, ISELAB, CVC-UAB"
__date__ = "10/01/2018"


class AttentionNet(nn.Module):
    """
    A simple 5conv 2fc CNN with painless attention
    """

    def __init__(self, attention_depth, attention_width, has_gates=True, reg_w=0.0):
        """

        Args:
            attention_depth: number of attention modules (starting by the last layer)
            attention_width: number of attention heads in each attention module
            has_gates: use gating (recommended)
            reg_w: inter-head regularization weight
        """
        super(AttentionNet, self).__init__()
        self.has_gates = has_gates
        self.orthogonal = reg_w
        self.attention_width = attention_width
        self.attention_depth = attention_depth
        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 32, 5, bias=False)  # 40 -> 18
        nn.init.kaiming_normal(self.conv1.weight.data)
        if self.attention_depth == 5:
            self.att1 = AttentionModule(32, 18, 18, nlabels=10, nheads=self.attention_width, has_gates=has_gates,
                                        reg_w=reg_w)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)  # 16 -> 8
        nn.init.kaiming_normal(self.conv2.weight.data)
        if self.attention_depth >= 4:
            self.att2 = AttentionModule(64, 8, 8, nlabels=10, nheads=self.attention_width, has_gates=has_gates,
                                        reg_w=reg_w)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, bias=False)  # 8 -> 6
        nn.init.kaiming_normal(self.conv3.weight.data)
        if self.attention_depth >= 3:
            self.att3 = AttentionModule(128, 6, 6, nlabels=10, nheads=self.attention_width, has_gates=has_gates,
                                        reg_w=reg_w)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, bias=False, padding=1)  # 6 -> 6
        nn.init.kaiming_normal(self.conv4.weight.data)
        if self.attention_depth >= 2:
            self.att4 = AttentionModule(128, 6, 6, nlabels=10, nheads=self.attention_width, has_gates=has_gates,
                                        reg_w=reg_w)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 3, bias=False)  # 6 -> 4
        nn.init.kaiming_normal(self.conv5.weight.data)
        if self.attention_depth >= 1:
            self.att5 = AttentionModule(128, 4, 4, nlabels=10, nheads=self.attention_width, has_gates=has_gates,
                                        reg_w=reg_w)
        self.bn6 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 ** 2, 256)
        nn.init.kaiming_normal(self.fc1.weight.data)
        self.bn7 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)
        nn.init.kaiming_normal(self.fc2.weight.data)
        # Gates for the output layer
        if has_gates:
            self.g_out = nn.Linear(256, 1, bias=False)
            self.g_bn = nn.BatchNorm1d(1)
        nn.init.kaiming_normal(self.g_out.weight.data)

    def reg_loss(self):
        """ Regularization loss

        Returns: a Variable containing the total weighted inter-head regularization loss

        """
        loss = 0
        for i in range(5, self.attention_depth, -1):
            loss += self.__getattr__("att%d" % i).reg_loss()
        return loss / self.attention_depth

    def forward(self, x):
        """ Pytorch forward function

        Args:
            x: input images

        Returns: log_softmax(logits)

        """
        outputs = []
        gates = []
        x = self.bn1(x)
        x = self.bn2(F.relu(F.max_pool2d(self.conv1(x), 2)))
        if self.attention_depth == 5:
            out1, g1 = self.att1(x)
            outputs.append(out1)
            gates.append(g1)
        x = self.bn3(F.relu(F.max_pool2d(self.conv2(x), 2)))
        if self.attention_depth >= 4:
            out2, g2 = self.att2(x)
            outputs.append(out2)
            gates.append(g2)
        x = self.bn4(F.relu(self.conv3(x)))
        if self.attention_depth >= 3:
            out3, g3 = self.att3(x)
            outputs.append(out3)
            gates.append(g3)
        x = self.bn5(F.relu(self.conv4(x)))
        if self.attention_depth >= 2:
            out4, g4 = self.att4(x)
            outputs.append(out4)
            gates.append(g4)
        x = self.bn6(F.relu(self.conv5(x)))
        if self.attention_depth >= 1:
            out5, g5 = self.att5(x)
            outputs.append(out5)
            gates.append(g5)
        x = x.view(x.size(0), 128 * 4 ** 2)
        x = self.bn7(F.relu(self.fc1(x)))
        output = F.log_softmax(self.fc2(x), 1)
        outputs.append(output.view(x.size(0), 1, 10))
        output = torch.cat(outputs, 1)
        if self.has_gates:
            g_out = F.tanh(self.g_bn(self.g_out(x)))
            gates.append(g_out)
            gates = torch.cat(gates, 1)
            gates = F.softmax(gates, 1).view(x.size(0), -1, 1)
            output = (output * gates).sum(1)
        else:
            output = output.mean(1)
        return output
