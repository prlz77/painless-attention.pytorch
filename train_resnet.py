import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from datasets.listfile import ImageList
from opts import parser
from prlz77_utils.loggers.json_logger import JsonLogger
from prlz77_utils.monitors.meter import Meter

__author__ = "prlz77, ISELAB, CVC-UAB"
__date__ = "10/01/2018"


def main(args):
    train_loss_meter = Meter()
    val_loss_meter = Meter()
    val_accuracy_meter = Meter()
    log = JsonLogger(args.log_path, rand_folder=True)
    log.update(args.__dict__)
    state = args.__dict__
    state['exp_dir'] = os.path.dirname(log.path)
    print(state)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_dataset = ImageList(args.root_folder, args.train_listfile,
                        transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(imagenet_mean, imagenet_std)
                        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, pin_memory=False, num_workers=args.num_workers)
    val_dataset = ImageList(args.root_folder, args.val_listfile,
                        transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(imagenet_mean, imagenet_std)
                        ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             pin_memory=False, num_workers=args.num_workers)

    if args.attention_depth == 0:
        from models.wide_resnet import WideResNet
        model = WideResNet().finetune(args.nlabels).cuda()
    else:
        from models.wide_resnet_attention import WideResNetAttention
        model = WideResNetAttention(args.nlabels, args.attention_depth, args.attention_width, args.has_gates,
                                  args.reg_weight).finetune(args.nlabels)

    # if args.load != "":
    #     net.load_state_dict(torch.load(args.load), strict=False)
    #     net = net.cuda()

    optimizer = optim.SGD([{'params': model.get_base_params(), 'lr': args.lr * 0.1},
                           {'params': model.get_classifier_params()}],
                          lr=args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)

    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, range(args.ngpu)).cuda()
        criterion = torch.nn.DataParallel(torch.nn.NLLLoss()).cuda()
    else:
        model = model.cuda()
        criterion = torch.nn.NLLLoss().cuda()


    def train():
        """

        """
        model.train()
        for data, label in train_loader:
            data, label = torch.autograd.Variable(data, requires_grad=False).cuda(async=True), \
                          torch.autograd.Variable(label, requires_grad=False).cuda()
            optimizer.zero_grad()
            if args.attention_depth > 0:
                output, loss = model(data)
                if args.reg_weight > 0:
                    loss = loss.mean()
                else:
                    loss = 0
            else:
                loss = 0
                output = model(data)
            loss += criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss_meter.update(loss.data[0], data.size(0))
        state['train_loss'] = train_loss_meter.mean()

    def val():
        """

        """
        model.eval()
        for data, label in val_loader:
            data, label = torch.autograd.Variable(data, volatile=True).cuda(async=True), \
                          torch.autograd.Variable(label, volatile=True).cuda()
            if args.attention_depth > 0:
                output, loss = model(data)
            else:
                output = model(data)
            loss = criterion(output, label)
            val_loss_meter.update(loss.data[0], data.size(0))
            preds = output.max(1)[1]
            val_accuracy_meter.update((preds == label).float().sum().data[0], data.size(0))
        state['val_loss'] = val_loss_meter.mean()
        state['val_accuracy'] = val_accuracy_meter.mean()

    best_accuracy = 0
    for epoch in range(args.epochs):
        train()
        val()
        if state['val_accuracy'] > best_accuracy:
            best_accuracy = state['val_accuracy']
            if args.save:
                torch.save(model.state_dict(), os.path.join(state["exp_dir"], "model.pytorch"))
        state['epoch'] = epoch + 1
        log.update(state)
        print(state)
        if (epoch + 1) in args.schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
