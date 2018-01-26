from models.wide_resnet import *
from modules.attention import AttentionModule, Gate

class WideResNetAttention(WideResNet):
    def __init__(self, nlabels, attention_depth=0, nheads=1, has_gates=True, reg_w=0.0, pretrain_path="pretrained/wrn-50-2.t7"):
        super(WideResNetAttention, self).__init__(pretrain_path)
        self.attention_outputs = []
        self.attention_depth = attention_depth
        self.has_gates = has_gates
        self.reg_w = reg_w

        for i in range(attention_depth):
            att = AttentionModule(2048 // (2**i), int(7 * 2**i), int(7*2**i), nlabels, nheads, has_gates, reg_w)
            self.__setattr__("att%d" %(3-i), att)

        if self.has_gates:
            self.output_gate = Gate(2048)

        # self.finetune(nlabels)

    def get_classifier_params(self):
        params = []
        for i in range(self.attention_depth):
            params += list(self.__getattr__("att%i" %(3-i)).parameters())
        if self.has_gates:
            params += list(self.output_gate.parameters())

        return params + list(self.linear.parameters())


    def reg_loss(self):
        loss = 0
        for i in range(self.attention_depth):
            loss += self.__getattr__("att%i" %(3-i)).reg_loss()
        return loss / self.attention_depth

    def forward(self, x):
        self.attention_outputs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2, 1)
        outputs = []
        gates = []
        for i in range(4):
            group = self.__getattr__("group%d" %i)
            x = group(x)
            if self.attention_depth > (3 - i):
                o, g = self.__getattr__("att%d" % i)(x)
                outputs += o
                if self.has_gates:
                    gates += g
        x = x.mean(3).mean(2)
        last_output = self.linear(x)
        if self.has_gates:
            last_gate = self.output_gate(x)
        else:
            last_gate = None

        if self.reg_w > 0:
            reg_loss = self.reg_loss()
        else:
            reg_loss = None

        return AttentionModule.aggregate(outputs, gates, last_output, last_gate), reg_loss

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
