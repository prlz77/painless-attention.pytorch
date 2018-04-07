# -*- coding: utf-8 -*-
"""
Loads a Wide Residual Network Pretrained on The Imagenet. Based on:
https://github.com/szagoruyko/wide-residual-networks

Requires the original lua pretrained model.
"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1 at gmail.com"

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.serialization import load_lua


def bn_layer(loaded):
    bn = nn.BatchNorm2d(loaded.weight.size(0))
    bn.weight.data = loaded.weight
    bn.bias.data = loaded.bias
    bn.running_mean = loaded.running_mean
    bn.running_var = loaded.running_var
    return bn


def conv_layer(loaded):
    conv = nn.Conv2d(loaded.weight.size(1), loaded.weight.size(0), loaded.weight.size(2),
                     stride=loaded.stride[0], padding=loaded.pad[0], bias=False)
    conv.weight.data = loaded.weight
    conv.groups = loaded.groups
    return conv


def linear_layer(loaded):
    linear = nn.Linear(loaded.weight.size(1), loaded.weight.size(0))
    linear.weight.data = loaded.weight
    linear.bias.data = loaded.bias
    return linear


class BasicBlock(nn.Module):
    def __init__(self, loaded):
        super(BasicBlock, self).__init__()
        loaded_block = loaded.modules[0].modules

        self.conv1 = conv_layer(loaded_block[0])
        self.bn1 = bn_layer(loaded_block[1])
        self.conv2 = conv_layer(loaded_block[3])
        self.bn2 = bn_layer(loaded_block[4])
        self.conv3 = conv_layer(loaded_block[6])
        self.bn3 = bn_layer(loaded_block[7])

        loaded_shortcut = loaded.modules[1].listModules()
        self.shortcut = len(loaded_shortcut) > 1
        if self.shortcut:
            self.shortcut_conv = conv_layer(loaded.modules[1].modules[0])
            self.shortcut_bn = bn_layer(loaded.modules[1].modules[1])

    def forward(self, x):
        x_block = self.conv1(x)
        x_block = self.bn1(x_block)
        x_block = F.relu(x_block)
        x_block = self.conv2(x_block)
        x_block = self.bn2(x_block)
        x_block = F.relu(x_block)
        x_block = self.conv3(x_block)
        x_block = self.bn3(x_block)
        if self.shortcut:
            x_shortcut = self.shortcut_conv(x)
            x_shortcut = self.shortcut_bn(x_shortcut)
        else:
            x_shortcut = x
        return F.relu(x_shortcut + x_block)


class Group(nn.Module):
    def __init__(self, loaded):
        super(Group, self).__init__()
        layer = []
        for i in range(len(loaded.modules)):
            layer.append(BasicBlock(loaded.modules[i].modules[0]))
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, pretrain_path='pretrained/wrn-50-2.t7'):
        super(WideResNet, self).__init__()

        if not os.path.isfile(pretrain_path):
            raise IOError("""wrn-50-2.t7 pre-trained model not found. 
            Please, download it from: https://yadi.sk/d/-8AWymOPyVZns and
            copy it into pretrained/wrn-50-2.t7""")

        loaded = load_lua(pretrain_path)
        self.conv1 = conv_layer(loaded.modules[0])
        self.bn1 = bn_layer(loaded.modules[1])
        self.group0 = Group(loaded.modules[4])
        self.group1 = Group(loaded.modules[5])
        self.group2 = Group(loaded.modules[6])
        self.group3 = Group(loaded.modules[7])
        self.linear = linear_layer(loaded.modules[10])

    def finetune(self, nlabels):
        self.nlabels = nlabels
        self.linear = nn.Linear(2048, nlabels)
        init.kaiming_normal(self.linear.weight.data)
        return self

    def get_base_params(self):
        params = []
        params += list(self.conv1.parameters())
        params += list(self.bn1.parameters())
        params += list(self.group0.parameters())
        params += list(self.group1.parameters())
        params += list(self.group2.parameters())
        params += list(self.group3.parameters())
        return params

    def get_classifier_params(self):
        return self.linear.parameters()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2, 1)
        x = self.group0(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        return F.log_softmax(self.linear(x.mean(3).mean(2)), dim=1)


# Test utility
if __name__ == '__main__':
    net = WideResNet().eval()
    import pylab
    import cv2
    import numpy as np

    cat = pylab.imread('demo/goldfish.jpeg') / 255.
    cat = cv2.resize(cat, (224, 224))
    cat = cat.transpose(2, 0, 1)
    cat -= np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    cat /= np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    cat = torch.Tensor(cat[None, ...])
    out = net(Variable(cat))
    max, ind = torch.max(net.linear(out), 1)
    # print(max)
    print(ind)
