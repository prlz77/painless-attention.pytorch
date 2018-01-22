from models.wide_resnet import *
from modules.attention import AttentionModule, Gate

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

class WideResNetAttention(WideResNet):
    def __init__(self, nlabels, attention_depth=0, nheads=1, has_gates=True, reg_w=0.0, pretrain_path="pretrained/wrn-50-2.t7"):
        super(WideResNetAttention, self).__init__(pretrain_path)
        self.attention_outputs = []
        self.has_gates = has_gates
        for i in range(attention_depth):
            group = self.__getattr__("group%d" %(3-i))
            att = AttentionModule(int(1024 / (2**i)), int(7 * 2**i), int(7*2**i), nlabels, nheads, has_gates, reg_w)
            self.__setattr__("att_%d" %(3-i), att)

        if self.has_gates:
            self.output_gate = Gate(2048)

        # self.finetune(nlabels)

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
        self.attention_outputs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2, 1)
        x = self.group0(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.mean(2).view(x.size(0), x.size(1))
        last_output = self.linear(x)
        if self.has_gates:
            last_gate = self.output_gate(x)
        else:
            last_gate = None

        return AttentionModule.aggregate(last_output, last_gate)

if __name__ == '__main__':
    import time
    t = time.time()
    net = WideResNetAttention(1000, attention_depth=3, nheads=4, has_gates=True, pretrain_path="pretrained/wrn-50-2.t7").eval()
    print(time.time() - t)
    import pylab
    import cv2
    import numpy as np
    im = pylab.imread('./demo/goldfish.jpeg')/255.
    im = cv2.resize(im, (224,224))
    im = im.transpose(2,0,1)
    im -= np.array([0.485, 0.456, 0.406]).reshape((3,1,1))
    im /= np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
    im = torch.Tensor(im[None,...])
    t = time.time()
    out = net(Variable(im))
    print(time.time() - t)
    max, ind = torch.max(out, 1)
    # max, ind = torch.max(net.linear(out), 1)
    # print(max)
    print(ind)
