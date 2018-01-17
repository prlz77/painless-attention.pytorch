import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from datasets.npyloader import NpyDataset
from models.attention_net import AttentionNet
from models.baseline import Baseline
from modules.stn import STN
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

    if args.train_distractors == 0:
        dataset = datasets.MNIST(args.mnist_path, train=True,
                                 transform=transforms.Compose([
                                     transforms.RandomCrop(40, padding=14),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.1307], [0.3081])
                                 ]))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=args.num_workers)
        dataset = datasets.MNIST(args.mnist_path, train=False,
                                 transform=transforms.Compose([
                                     transforms.RandomCrop(40, padding=14),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.1307], [0.3081])
                                 ]))
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                                 pin_memory=True, num_workers=args.num_workers)
    elif args.train_distractors > 0:
        train_dataset = NpyDataset(os.path.join(args.mnist_path, 'train_%d.npy' % args.test_distractors))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=args.num_workers)
        val_dataset = NpyDataset(os.path.join(args.mnist_path, 'valid_%d.npy' % args.train_distractors))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                 pin_memory=True, num_workers=args.num_workers)
        test_dataset = NpyDataset(os.path.join(args.mnist_path, 'test_%d.npy' % args.test_distractors))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  pin_memory=True, num_workers=args.num_workers)
    else:
        raise ValueError("#train distractors not >= 0")

    if args.model == "baseline":
        model = Baseline().cuda()
    elif args.model == 'stn':
        stn = STN().cuda()
        baseline = Baseline().cuda()
        model = torch.nn.Sequential(stn, baseline)
    else:
        model = AttentionNet(args.att_depth, args.nheads, args.has_gates, args.reg_w).cuda()

    if args.load != "":
        model.load_state_dict(torch.load(args.load), strict=False)
        model = model.cuda()

    if args.model != "stn":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-5, momentum=0.9)
    else:
        optimizer = optim.SGD([{'params': baseline.parameters()},
                               {'params': stn.parameters(), 'lr': 0.001}],
                              lr=args.learning_rate, weight_decay=1e-5, momentum=0.9)

    def train():
        """

        """
        model.train()
        for data, label in train_loader:
            data, label = torch.autograd.Variable(data, requires_grad=False).cuda(), \
                          torch.autograd.Variable(label, requires_grad=False).cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, label)
            if args.reg_w > 0:
                loss += model.reg_loss()
            loss.backward()
            optimizer.step()
            train_loss_meter.update(loss.data[0], data.size(0))
        state['train_loss'] = train_loss_meter.mean()

    def val():
        """

        """
        model.eval()
        for data, label in val_loader:
            data, label = torch.autograd.Variable(data, requires_grad=False).cuda(), \
                          torch.autograd.Variable(label, requires_grad=False).cuda()
            output = model(data)
            loss = F.nll_loss(output, label)
            val_loss_meter.update(loss.data[0], data.size(0))
            preds = output.max(1)[1]
            val_accuracy_meter.update((preds == label).float().sum().data[0], data.size(0))
        state['val_loss'] = val_loss_meter.mean()
        state['val_accuracy'] = val_accuracy_meter.mean()

    def test():
        """

        """
        model.eval()
        for data, label in test_loader:
            data, label = torch.autograd.Variable(data, requires_grad=False).cuda(), \
                          torch.autograd.Variable(label, requires_grad=False).cuda()
            output = model(data)
            loss = F.cross_entropy(output, label)
            val_loss_meter.update(loss.data[0], data.size(0))
            preds = output.max(1)[1]
            val_accuracy_meter.update((preds == label).float().sum().data[0], data.size(0))
        state['test_loss'] = val_loss_meter.mean()
        state['test_accuracy'] = val_accuracy_meter.mean()

    if args.load != "":
        test()
        print(state)
        log.update(state)
    else:
        for epoch in range(args.epochs):
            train()
            val()
            if epoch == args.epochs - 1:
                test()
            state['epoch'] = epoch + 1
            log.update(state)
            print(state)
            if (epoch + 1) in args.schedule:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
        if args.save:
            torch.save(model.state_dict(), os.path.join(state["exp_dir"], "model.pytorch"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mnist_path', type=str, help="mnist dataset path")
    parser.add_argument('--att_depth', type=int, default=1, help="attention depth")
    parser.add_argument('--batch_size', type=int, default=100, help="batch size")
    parser.add_argument('--epochs', '-e', type=int, default=100, help="number of training epochs")
    parser.add_argument('--has_gates', '-g', action="store_true", help="use attention gates")
    parser.add_argument('--learning_rate', '--lr', type=float, default=0.1, help="log path")
    parser.add_argument('--log_path', type=str, default="./logs", help="log path")
    parser.add_argument('--model', type=str, default="baseline", choices=["baseline", "attention", "stn"],
                        help="model type")
    parser.add_argument('--nheads', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2, help="Number of prefetch threads.")
    parser.add_argument('--reg_w', type=float, default=0, help="Weight to regularize attention masks.")
    parser.add_argument('--schedule', type=int, nargs='+', default=[60], help="learning rate decay schedule")
    parser.add_argument('--train_distractors', type=int, default=8, help="Number of distractors in the train set")
    parser.add_argument('--test_distractors', type=int, default=8, help="Number of distractors in the test set")
    parser.add_argument('--test_only', action='store_true', help="")
    parser.add_argument('--load', type=str, default="", help="load saved model")
    parser.add_argument('--save', action='store_true', help="save the model")
    main(parser.parse_args())
