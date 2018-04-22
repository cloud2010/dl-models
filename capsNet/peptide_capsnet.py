# -*- coding: utf-8 -*-
"""
A PyTorch implementation of CapsNet for peptide datasets only in CPU
Modified: liumin@shmtu.edu.cn
Reference: `https://github.com/XifengGuo/CapsNet-Pytorch`
"""
import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from torch import nn
from torch.optim import Adam, lr_scheduler
from sklearn.model_selection import KFold
from torch.autograd import Variable
from capsulelayers import DenseCapsule, PrimaryCapsule
from utils import bi_model_evaluation


class CapsuleNet(nn.Module):
    """
    A Capsule Network on peptide.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """

    def __init__(self, input_size, classes, routings):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=5, stride=1, padding=0)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(256, 256, 8, kernel_size=5, stride=2, padding=0)

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=32*6*6, in_dim_caps=8,
                                      out_num_caps=classes, out_dim_caps=16, routings=routings)

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(16*classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)
        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1), 1.))
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return length, reconstruction.view(-1, *self.input_size)


def caps_loss(y_true, y_pred, x, x_recon, lam_recon):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
    :param x_recon: reconstructed data, size is same as `x`
    :param lam_recon: coefficient for reconstruction loss
    :return: Variable contains a scalar loss value.
    """
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()

    L_recon = nn.MSELoss()(x_recon, x)

    return L_margin + lam_recon * L_recon


def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    for x, y in test_loader:
        x, y = Variable(x, volatile=True), Variable(y)
        y_pred, x_recon = model(x)
        test_loss += caps_loss(y, y_pred, x, x_recon,
                               args.lam_recon).data[0] * x.size(0)  # sum up batch loss
        y_pred = y_pred.data.max(1)[1]
        y_true = y.data.max(1)[1]
        correct += y_pred.eq(y_true).sum()

    test_loss /= len(test_loader.dataset)
    return test_loss, correct / len(test_loader.dataset)


def train(model, train_loader, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param train_loader: torch.utils.data.DataLoader for training data
    :param test_loader: torch.utils.data.DataLoader for test data
    :param args: arguments
    :return: The trained model
    """
    print('-'*6 + ' Begin Training ' + '-'*6)
    from time import time

    t0 = time()
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    best_val_acc = 0.
    for epoch in range(args.epochs):
        model.train()  # set to training mode
        lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
        ti = time()
        training_loss = 0.0
        for i, (x, y) in enumerate(train_loader):  # batch training
            x, y = Variable(x), Variable(y)
            optimizer.zero_grad()  # set gradients of optimizer to zero
            y_pred, x_recon = model(x, y)  # forward
            loss = caps_loss(y, y_pred, x, x_recon, args.lam_recon)  # compute loss
            loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
            training_loss += loss.data[0] * x.size(0)  # record the batch loss
            optimizer.step()  # update the trainable parameters with computed gradients

        # compute validation loss and acc
        # val_loss, val_acc = test(model, test_loader, args)
        val_loss, val_acc = 0., 100.
        
        print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
              % (epoch, training_loss / len(train_loader.dataset),
                 val_loss, val_acc, time() - ti))
        # if val_acc > best_val_acc:  # update best validation acc and save model
        #     best_val_acc = val_acc
        #     torch.save(model.state_dict(), os.path.join(args.savepath, '/epoch%d.pkl' % epoch))
        #     print("best val_acc increased to %.4f" % best_val_acc)
    
    torch.save(model.state_dict(), os.path.join(args.savepath, 'trained_model.pkl'))
    print('Trained model saved to \'%s/trained_model.pkl\'' % args.savepath)
    print('-'*6 + ' End Training ' + '-'*6)
    print("[Total time: {0:.6f} mins = {1:.6f} seconds]".format(((time() - t0) / 60), (time() - t0)))
    return model


def load_datasets(path, bs=100):
    """
    Construct dataloaders for peptide training and test data. Data augmentation is also done here.
    :param path: file path of the dataset
    :param bs: batch size
    :return: train_loader, test_loader
    """
    # kwargs = {'num_workers': 0, 'pin_memory': True}

    # FILE = "peptide_caps_net.csv"  # 数据集名称
    # DATASETS = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir, "datasets", FILE))  # 构建路径
    # print(os.getcwd())
    DATASETS = os.path.abspath(path)
    print("Dataset Name:", DATASETS)

    df = pd.read_csv(DATASETS)
    # 第1列为分类标注名称
    labels = df.values[:, 0].astype(np.int32)
    # 调整 label 2为1(N样本), 1为0(P样本), 便于转换为 One-hot Vector
    labels[labels == 1] = 0
    labels[labels == 2] = 1
    t_labels = torch.from_numpy(labels).view(-1, 1)
    # print(t_labels)
    # 转换为 One-hot Vector
    t_onehot = torch.zeros(labels.size, 2)
    t_onehot.scatter_(1, t_labels.type(torch.LongTensor), 1)
    # print(t_onehot)

    # 第3列之后为特征矩阵
    features = df.values[:, 2:].astype(np.int32)
    """
    返回一个内存连续的有相同数据的tensor，否则无法做reshape
    -Input:  (batch, channels, width, height), optional (batch, classes)
    -Output: ((batch, classes), (batch, channels, width, height))
    """
    t_features = torch.from_numpy(features).contiguous(
    ).view(-1, 1, 20, 20).type(torch.FloatTensor)  # 4-D Tensor

    # Using PyTorch DataLoader
    # 构建 pytorch 数据集
    train_dataset = Data.TensorDataset(data_tensor=t_features, target_tensor=t_onehot)
    # remove num_workers settings
    train_loader = Data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=bs)

    return train_loader

    # return train_loader, test_loader


if __name__ == "__main__":
    import argparse

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on Peptide datasets.")
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.0005 * 400, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")  # num_routing should > 0
    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of kfolds. Must be at least 2.", default=10)
    parser.add_argument('--datapath', required=True, help="Path of dataset.")
    parser.add_argument('--savepath', default='./result',
                        help="The directory for logs and summaries.")
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)

    # load data
    # train_loader, test_loader = load_datasets(args.datapath, batch_size=args.batch_size)
    train_datasets = load_datasets(args.datapath, bs=args.batch_size)

    # define training model
    # peptide 数据集为两分类问题，样本矩阵大小 20 * 20，即 data size = [1, 20, 20]
    model = CapsuleNet(input_size=[1, 20, 20], classes=2, routings=3)
    # model.cuda()
    # output CapsNet structure
    print(model)

    # start training
    train(model, train_datasets, args)

    # train and validate
    # 迭代输出DataLoader相关信息
    # for epoch in range(args.epochs):
    #     for step, (batch_x, batch_y) in enumerate(train_datasets):
    #         # print('Epoch: ', epoch, '| Step: ', step, '| Batch x: ', batch_x.numpy(), '| Batch y: ', batch_y.numpy())
    #         print('Epoch: ', epoch, '| Step: ', step, '| Batch y: ', batch_y.numpy())

    # train or test
    # if args.weights is not None:  # init the model weights with provided one
    #     model.load_state_dict(torch.load(args.weights))
    # if not args.testing:
    #     train(model, train_loader, test_loader, args)
    # else:  # testing
    #     if args.weights is None:
    #         print('No weights are provided. Will test using random initialized weights.')
    #     test_loss, test_acc = test(model=model, test_loader=test_loader, args=args)
    #     print('test acc = %.4f, test loss = %.5f' % (test_acc, test_loss))
    #     show_reconstruction(model, test_loader, 50, args)
