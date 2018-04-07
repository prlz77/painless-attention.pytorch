# -*- coding: utf-8 -*-
"""
Defines a Wide Attention Residual Network (WARN) for the Imagenet
"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1 at gmail.com"

from models.wide_resnet import *
from modules.attention import AttentionModule, Gate


class WideResNetAttention(WideResNet):
    """
    Wide Residual Attention Network (WARN). Inherits from the base Wide Residual Network
    """
    def __init__(self, nlabels, attention_depth=0, nheads=1, has_gates=True, reg_w=0.0,
                 pretrain_path="pretrained/wrn-50-2.t7", attention_output='all', attention_type='softmax'):
        """ Constructor

        Args:
            nlabels: number of output labels
            attention_depth: number groups to which append an attention module
            nheads: number of attention heads per attention module
            has_gates: whether to use gating
            reg_w: inter-head regularization factor
            pretrain_path: path with pretrained weights
            attention_output: attention mechanism to produce the final output
            attention_type: soft attention non-linearity
        """
        super(WideResNetAttention, self).__init__(pretrain_path)
        self.attention_outputs = []
        self.attention_depth = attention_depth
        self.has_gates = has_gates
        self.reg_w = reg_w
        self.attention_output = attention_output
        self.attention_type = attention_type

        for i in range(attention_depth):
            att = AttentionModule(2048 // (2 ** (i + 1)), nlabels, nheads, reg_w)
            self.__setattr__("att%d" % (2 - i), att)

        if self.has_gates:
            ngates = self.attention_depth
            if self.attention_output == 'all':
                ngates += 1

            self.output_gate = Gate(2048, ngates, gate_depth=1)
        # self.finetune(nlabels)

    def get_classifier_params(self):
        """ Helper function to obtain the parameters of the final classifier

        Returns: Parameters of the final classifier

        """
        params = []
        for i in range(self.attention_depth):
            params += list(self.__getattr__("att%i" % (2 - i)).parameters())
        if self.has_gates:
            params += list(self.output_gate.parameters())

        return params + list(self.linear.parameters())

    def reg_loss(self):
        """ Computes the total regularization loss from all the attention modules.

        Returns: regularization loss

        """
        loss = 0
        for i in range(self.attention_depth):
            loss += self.__getattr__("att%i" % (2 - i)).reg_loss()
        return loss / self.attention_depth

    def forward(self, x):
        """ Network forward function

        Args:
            x: input image

        Returns: tuple (final output with attention, regularization loss)

        """
        self.attention_outputs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2, 1)
        outputs = []
        for i in range(4):
            group = self.__getattr__("group%d" % i)
            x = group(x)
            if i > (2 - self.attention_depth) and i <= 2:
                outputs.append(self.__getattr__("att%d" % i)(x))
        x = x.mean(3).mean(2)
        if self.has_gates:
            gates = self.output_gate(x)
        else:
            gates = None

        if self.reg_w > 0:
            reg_loss = self.reg_loss()
        else:
            reg_loss = None

        if self.attention_output == 'all':
            outputs.append(self.linear(x).view(x.size(0), 1, -1))
            ret = AttentionModule.aggregate(outputs, gates, self.attention_type)
        elif self.attention_output == '50':
            ret = (AttentionModule.aggregate(outputs, gates, self.attention_type) + F.log_softmax(self.linear(x),
                                                                                                  dim=1)) / 2.
        else:
            raise ValueError(self.attention_output)
        return ret, reg_loss

