import torch
import torch.nn.functional as F
from modules.attention import AttentionModule, Gate

class Block(torch.nn.Module):
    def __init__(self, ni, no, stride, dropout, save_input=False):
        super(Block, self).__init__()
        self.dropout = dropout
        self.conv0 = torch.nn.Conv2d(ni, no, 3, stride=stride, padding=1, bias=False)
        torch.nn.init.kaiming_normal(self.conv0.weight.data)
        self.bn0 = torch.nn.BatchNorm2d(ni)
        self.conv1 = torch.nn.Conv2d(no, no, 3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(no)
        torch.nn.init.kaiming_normal(self.conv1.weight.data)
        self.reduce = ni != no
        self.save_input = save_input
        if self.reduce:
            self.conv_reduce = torch.nn.Conv2d(ni, no, 1, stride=stride, bias=False)
            torch.nn.init.kaiming_normal(self.conv_reduce.weight.data)

    def forward(self, x):
        block_input = F.relu(self.bn0(x), True)

        if self.save_input:
            self.block_input = block_input

        y = self.conv0(block_input)
        o2 = F.relu(self.bn1(y), inplace=True)
        if self.dropout > 0:
            o2 = F.dropout2d(o2, self.dropout, training=self.training, inplace=True)
        z = self.conv1(o2)
        if self.reduce:
            return z + self.conv_reduce(x)
        else:
            return z + x

class Group(torch.nn.Module):
    def __init__(self, ni, no, n, stride, dropout):
        super(Group, self).__init__()
        self.n = n
        for i in range(n):
            self.__setattr__("block_%d" %i, Block(ni if i == 0 else no, no, stride if i == 0 else 1, dropout, save_input=(i==0)))

    def forward(self, x):
        for i in range(self.n):
            x = self.__getattr__("block_%d" %i)(x)
        return x


class WideResNetAttention(torch.nn.Module):
    def __init__(self, depth, width, num_classes, dropout, attention_depth, attention_width, reg_w=0, attention_output="all", attention_type="softmax"):
        super(WideResNetAttention, self).__init__()
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        self.n = (depth - 4) // 6
        self.num_classes = num_classes
        widths = torch.Tensor([16, 32, 64]).mul(width).int()
        self.conv0 = torch.nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(16)
        torch.nn.init.kaiming_normal(self.conv0.weight.data)
        self.group_0 = Group(16, widths[0], self.n, 1, dropout)
        self.group_1 = Group(widths[0], widths[1], self.n, 2, dropout)
        self.group_2 = Group(widths[1], widths[2], self.n, 2, dropout)
        self.bn_g2 = torch.nn.BatchNorm2d(widths[2])
        self.classifier = torch.nn.Linear(widths[2], self.num_classes)

        self.attention_depth = attention_depth
        self.attention_width = attention_width
        self.reg_w = reg_w
        self.attention_output = attention_output
        self.attention_type = attention_type

        self.attention_layers = [2 - i for i in range(self.attention_depth)]
        print("Attention after groups %s" %(str(self.attention_layers)))
        for i in self.attention_layers:
            att = AttentionModule(widths[i], num_classes, attention_width, reg_w)
            self.__setattr__("att%d" %(i), att)

        ngates = self.attention_depth
        if self.attention_output == 'all':
            ngates += 1

        self.output_gate = Gate(widths[-1], ngates, gate_depth=1)

    def reg_loss(self):
        loss = 0
        for i in range(self.attention_depth):
            loss += self.__getattr__("att%i" % self.attention_layers[i]).reg_loss()
        return loss / self.attention_depth

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)), True)
        group0 = self.group_0(x)
        group1 = self.group_1(group0)
        group2 = F.relu(self.bn_g2(self.group_2(group1), True))

        groups = [group1.block_0.block_input, group2.block_0.block_input, group2]
        attention_outputs = []

        for i in self.attention_layers:
            attention_outputs.append(self.__getattr__("att%d" %i)(groups[i]))

        o = F.avg_pool2d(group2, 8, 1, 0)
        o = o.view(o.size(0), -1)

        gates = self.output_gate(o)

        if self.training and self.reg_w > 0:
            reg_loss = self.reg_loss()
        else:
            reg_loss = None

        if self.attention_output == "all":
            attention_outputs.append(self.classifier(o).view(o.size(0), 1, -1))
            ret = AttentionModule.aggregate(attention_outputs, gates, self.attention_type)
        elif self.attention_output == '50':
            ret = (AttentionModule.aggregate(attention_outputs, gates, self.attention_type) + F.log_softmax(self.classifier(o), dim=1)) / 2.
        else:
            raise ValueError(self.attention_output)
        return ret, reg_loss


