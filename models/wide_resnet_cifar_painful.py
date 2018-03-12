import torch
import torch.nn.functional as F
from modules.attention import AttentionModule, Gate

class Block(torch.nn.Module):
    def __init__(self, ni, no, stride):
        super(Block, self).__init__()
        self.bn0 = torch.nn.BatchNorm2d(ni)
        self.conv0 = torch.nn.Conv2d(ni, no, 3, stride=stride, padding=1, bias=False)
        torch.nn.init.kaiming_normal(self.conv0.weight.data)
        self.bn1 = torch.nn.BatchNorm2d(no)
        self.conv1 = torch.nn.Conv2d(no, no, 3, stride=1, padding=1, bias=False)
        torch.nn.init.kaiming_normal(self.conv1.weight.data)
        self.reduce = ni != no
        if self.reduce:
            self.conv_reduce = torch.nn.Conv2d(ni, no, 1, stride=stride, bias=False)
            torch.nn.init.kaiming_normal(self.conv_reduce.weight.data)

    def forward(self, x):
        o1 = F.relu(self.bn0(x), inplace=True)
        y = self.conv0(o1)
        o2 = F.relu(self.bn1(y), inplace=True)
        z = self.conv1(o2)
        if self.reduce:
            return z + self.conv_reduce(o1)
        else:
            return z + x

class Group(torch.nn.Module):
    def __init__(self, ni, no, n, stride):
        super(Group, self).__init__()
        self.n = n
        for i in range(n):
            self.__setattr__("block_%d" %i, Block(ni if i == 0 else no, no, stride if i == 0 else 1))

    def forward(self, x):
        for i in range(self.n):
            x = self.__getattr__("block_%d" %i)(x)
        return x

class PainfulAttention(torch.nn.Module):
    def __init__(self, local_ch, global_ch):
        super(PainfulAttention, self).__init__()
        self.conv_reduce = torch.nn.Conv2d(local_ch + global_ch, 1, 1, padding=0, bias=False)
        torch.nn.init.kaiming_normal(self.conv_reduce.weight.data)
        self.bn = torch.nn.BatchNorm2d(local_ch)


    def forward(self, local_feat, global_feat):
        local_feat = F.relu(self.bn(local_feat), True)
        local_feat = F.dropout2d(local_feat, 0.5, self.training, True)
        bs, ch, h, w = local_feat.size()
        global_feat = global_feat.resize(bs, global_feat.size(1), 1, 1)
        global_feat = global_feat.expand(bs, global_feat.size(1), h, w)
        cat_feat = torch.cat([local_feat, global_feat], 1)
        mask = F.softmax(self.conv_reduce(cat_feat).resize(bs, 1, h*w), dim=2)
        return (local_feat * mask.resize(bs, 1, h, w)).sum(3).sum(2)


class WideResNetAttention(torch.nn.Module):
    def __init__(self, depth, width, num_classes,  attention_depth):
        super(WideResNetAttention, self).__init__()
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        self.n = (depth - 4) // 6
        self.num_classes = num_classes
        widths = torch.Tensor([16, 32, 64]).mul(width).int()
        self.conv0 = torch.nn.Conv2d(3, 16, 3, padding=1, bias=False)
        torch.nn.init.kaiming_normal(self.conv0.weight.data)
        self.group_0 = Group(16, widths[0], self.n, 1)
        self.group_1 = Group(widths[0], widths[1], self.n, 2)
        self.group_2 = Group(widths[1], widths[2], self.n, 2)
        self.extra_bn_1 = torch.nn.BatchNorm2d(widths[2])
        self.extra_conv_1 = torch.nn.Conv2d(widths[2], widths[2], 3, padding=1, bias=False)
        torch.nn.init.kaiming_normal(self.extra_conv_1.weight.data)
        self.extra_bn_2 = torch.nn.BatchNorm2d(widths[2])
        self.extra_conv_2 = torch.nn.Conv2d(widths[2], widths[2], 3, padding=1, bias=False)
        torch.nn.init.kaiming_normal(self.extra_conv_2.weight.data)
        self.global_features = torch.nn.Linear(widths[2], widths[2], bias=False)
        torch.nn.init.kaiming_normal(self.global_features.weight.data)

        self.attention_depth = attention_depth
        self.attention_layers = [2 - i for i in range(self.attention_depth)]
        print("Attention after groups %s" %(str(self.attention_layers)))
        for i in self.attention_layers:
            att = PainfulAttention(widths[i], widths[-1])
            self.__setattr__("att%d" %(i), att)

        self.final_width = sum([widths[i] for i in self.attention_layers])
        self.final_batchnorm = torch.nn.BatchNorm2d(self.final_width)
        self.classifier = torch.nn.Linear(self.final_width, num_classes)
        torch.nn.init.kaiming_normal(self.classifier.weight.data)


    def reg_loss(self):
        loss = 0
        for i in range(self.attention_depth):
            loss += self.__getattr__("att%i" % self.attention_layers[i]).reg_loss()
        return loss / self.attention_depth

    def forward(self, x):
        x = self.conv0(x)
        group0 = self.group_0(x)
        group1 = self.group_1(group0)
        group2 = self.group_2(group1)
        group2 = self.extra_conv_1(group2)
        group2 = self.extra_conv_2(group2)

        global_features = F.relu(self.extra_bn_1(group2))
        global_features = F.max_pool2d(global_features, 2, 2, 0)
        global_features = F.relu(self.extra_bn_2(global_features))
        global_features = F.dropout2d(global_features, 0.4, training=self.training)
        global_features = F.max_pool2d(global_features, 4, 4, 0)
        global_features = global_features.view(global_features.size(0), -1)
        global_features = F.dropout(global_features, training=self.training)
        global_features = self.global_features(global_features)

        final_features = []
        if 1 in self.attention_layers:
            final_features.append(self.att1(group1, global_features))

        if 2 in self.attention_layers:
            final_features.append(self.att2(group2, global_features))

        final_features = torch.cat(final_features, 1)
        final_features = F.relu(self.final_batchnorm(final_features), True)
        return self.classifier(final_features)
