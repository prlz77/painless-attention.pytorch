import numpy as np
import torch
import torch.nn.functional as F


class OutputHead(torch.nn.Module):
    def __init__(self, in_ch, h, w, nlabels, has_gates):
        super(OutputHead, self).__init__()
        self.in_ch = in_ch
        self.h = h
        self.w = w
        self.nlabels = nlabels
        self.has_gates = has_gates

        self.F = torch.nn.Linear(in_ch * h * w, in_ch, bias=False)
        torch.nn.init.kaiming_normal(self.F.weight.data)
        self.bn = torch.nn.BatchNorm1d(in_ch)
        self.out = torch.nn.Linear(in_ch, nlabels)
        torch.nn.init.kaiming_normal(self.out.weight.data)

        if has_gates:
            self.gate = torch.nn.Linear(in_ch, 1, bias=False)
            self.bn_gate = torch.nn.BatchNorm1d(1)
            torch.nn.init.kaiming_normal(self.gate.weight.data)

    def forward(self, x):
        x = self.bn(F.relu(self.F(x.contiguous().view(x.size(0), -1))))
        return self.out(x), F.tanh(self.bn_gate(self.gate(x))) if self.has_gates else None


class AttentionHead(torch.nn.Module):
    def __init__(self, in_ch, nheads=1, has_gates=True):
        super(AttentionHead, self).__init__()
        self.nheads = nheads
        self.conv = torch.nn.Conv2d(in_ch, nheads, kernel_size=3, padding=1, bias=False)
        torch.nn.init.kaiming_normal(self.conv.weight.data)
        self.bn = torch.nn.BatchNorm2d(in_ch)
        self.register_buffer("diag",
                             torch.from_numpy(
                                 1 - np.eye(self.nheads, self.nheads).reshape(1, self.nheads, self.nheads)).float())

    def reg_loss(self):
        mask2loss = self.att_mask.view(self.att_mask.size(0), self.nheads, -1)
        reg_loss = torch.bmm(mask2loss, mask2loss.transpose(1, 2)) * torch.autograd.Variable(self.diag,
                                                                                             requires_grad=False)
        return (reg_loss.view(reg_loss.size(0), -1) ** 2).mean(1).mean(0)

    def forward(self, x):
        b, c, h, w = x.size()
        self.att_mask = F.softmax(self.conv(x).view(b, self.nheads * w * h), 1).view(b, self.nheads, h, w)
        xatt_mask = x.view(b, 1, c, h, w) * self.att_mask.view(b, self.nheads, 1, h, w)
        return self.bn(xatt_mask.view(b * self.nheads, c, h * w))


class AttentionModule(torch.nn.Module):
    def __init__(self, in_ch, h, w, nlabels, nheads=1, has_gates=True, reg_w=0.0):
        super(AttentionModule, self).__init__()
        self.nlabels = nlabels
        self.nheads = nheads
        self.has_gates = has_gates
        self.atthead = AttentionHead(in_ch, nheads, has_gates)
        self.reg_w = reg_w
        for i in range(self.nheads):
            self.__setattr__('outhead_%d' % i, OutputHead(in_ch, h, w, nlabels, has_gates))

    def reg_loss(self):
        return self.atthead.reg_loss() * self.reg_w

    def forward(self, x):
        b, c, h, w = x.size()
        atthead = self.atthead(x).view(b, self.nheads, c, h, w)

        output = []
        if self.has_gates:
            gates = []
        else:
            gates = None

        for i in range(self.nheads):
            outhead = self.__getattr__("outhead_%d" % i)
            out, g = outhead(atthead[:, i, ...])
            output.append(out.view(b, 1, self.nlabels))
            if self.has_gates:
                gates.append(g.view(b, 1))
        output = torch.cat(output, 1)

        if self.has_gates:
            gates = torch.cat(gates, 1)

        output = F.log_softmax(output, dim=2)

        return output, gates
