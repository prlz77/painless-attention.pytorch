import numpy as np
import torch
import torch.nn.functional as F

__author__ = "prlz77, ISELAB, CVC-UAB"
__date__ = "10/01/2018"


class Gate(torch.nn.Module):
    """
    Attention Gate. Weights Attention output by its importance.
    """
    def __init__(self, in_ch, ngates=1):
        """ Constructor

        Args:
            in_ch: number of input channels.
        """
        super(Gate, self).__init__()
        self.bn = torch.nn.BatchNorm1d(1)
        self.gates = torch.nn.Linear(in_ch, ngates, bias=False)
        torch.nn.init.kaiming_normal(self.gates.weight.data)

    def forward(self, x):
        """ Pytorch forward function

        Args:
            x: input Variable

        Returns: gate value (Variable)

        """
        return F.tanh(self.bn(self.gates(x)))


class AttentionHead(torch.nn.Module):
    """ Attention Heads

    Attentds a given feature map. Provides inter-mask regularization.
    """

    def __init__(self, in_ch, nheads=1):
        """ Constructor

        Args:
            in_ch: input feature map channels
            nheads: number of attention masks
        """
        super(AttentionHead, self).__init__()
        self.nheads = nheads
        self.conv = torch.nn.Conv2d(in_ch, nheads, kernel_size=3, padding=1, bias=False)
        torch.nn.init.kaiming_normal(self.conv.weight.data)
        self.register_buffer("diag",
                             torch.from_numpy(
                                 1 - np.eye(self.nheads, self.nheads).reshape(1, self.nheads, self.nheads)).float())

    def reg_loss(self):
        """ Regularization Loss

        Returns: a Variable with the inter-head regularization loss.

        """
        mask2loss = self.att_mask.view(self.att_mask.size(0), self.nheads, -1)
        reg_loss = torch.bmm(mask2loss, mask2loss.transpose(1, 2)) * torch.autograd.Variable(self.diag,
                                                                                             requires_grad=False)
        return (reg_loss.view(reg_loss.size(0), -1) ** 2).mean(1).mean(0)

    def forward(self, x):
        """ Pytorch Forward

        Args:
            x: input feature map

        Returns: the multiple attended feature maps

        """
        b, c, h, w = x.size()
        self.att_mask = F.softmax(self.conv(x).view(b, self.nheads, w * h), 2).view(b, self.nheads, h, w)
        return self.att_mask

class OutHead(torch.nn.Module):
    """ Attention Heads

    Attentds a given feature map. Provides inter-mask regularization.
    """

    def __init__(self, in_ch, out_ch):
        """ Constructor

        Args:
            in_ch: input feature map channels
            nheads: number of attention masks
        """
        super(OutHead, self).__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        torch.nn.init.kaiming_normal(self.conv.weight.data)
        self.bn = torch.nn.BatchNorm2d(out_ch)

    def forward(self, x):
        """ Pytorch Forward

        Args:
            x: input feature map

        Returns: the multiple attended feature maps

        """
        return self.bn(self.conv(x))

class AttentionModule(torch.nn.Module):
    """ Attention Module

    Applies different attention masks with the Attention Heads and ouputs classification hypotheses.
    """

    def __init__(self, in_ch, h, w, nlabels, nheads=1, reg_w=0.0):
        """ Constructor

        Args:
            in_ch: number of input feature map channels
            h: input feature map height
            w: input feature map width
            nlabels: number of output classes
            nheads: number of attention heads
            has_gates: whether to use gating (recommended)
            reg_w: inter-mask regularization weight
        """
        super(AttentionModule, self).__init__()
        self.in_ch = in_ch
        self.nlabels = nlabels
        self.nheads = nheads
        self.reg_w = reg_w

        self.att_head = AttentionHead(in_ch, nheads)
        self.out_head = OutHead(in_ch, nlabels*nheads)


    def reg_loss(self):
        """ Regularization loss

        Returns: A Variable with the inter-mask regularization loss for this  Attention Module.

        """
        return self.atthead.reg_loss() * self.reg_w

    def forward(self, x):
        """ Pytorch Forward

        Args:
            x: input feature map.

        Returns: tuple with predictions and gates. Gets are set to None if disabled.

        """
        b, c, h, w = x.size()
        att_mask = self.att_head(x)
        output = self.out_head(x.view(b, 1, c, h, w)) * att_mask.view(b, self.nheads, 1, h, w)
        output = (output.view(x.size(0), 1, x.size(1), x.size(2)*x.size(3)) * att_mask).sum(3)
        return output

    @staticmethod
    def aggregate(outputs, gates, last_output, last_gate=None):
        """ Generates the final output after aggregating all the attention modules.

        Args:
            last_output: network output
            last_gate: gate for the network output

        Returns: final network prediction

        """
        outputs.append(last_output.view(last_output.size(0), 1, -1))
        outputs = torch.cat(outputs, 1)
        outputs = F.log_softmax(outputs, dim=2)
        if last_gate is not None:
            gates.append(last_gate)
            gates = torch.cat(gates, 1)
            gates = F.softmax(gates, 1).view(gates.size(0), -1, 1)
            ret = (outputs * gates).sum(1)
        else:
            ret = outputs.mean(1)

        return ret
