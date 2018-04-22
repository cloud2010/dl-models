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

from torch.autograd import Variable
from capsulelayers import DenseCapsule, PrimaryCapsule



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
            index = length.data.max(dim=1)[1]
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
    return test_loss, correct / len(test_loader.dataset), y_pred, y_true


def train(model, train_loader, test_loader, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param train_loader: torch.utils.data.DataLoader for training data
    :param test_loader: torch.utils.data.DataLoader for test data
    :param args: arguments
    :return: The test predicted and true values
    """
    print('-'*6 + ' Begin Training ' + '-'*6)
    from time import time

    t0 = time()
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    best_test_acc = 0.
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

        # compute test datasets loss and acc
        test_loss, test_acc, test_pred, test_true = test(model, test_loader, args)

        print("\n==> Epoch {0}: train_loss={1:.5f}, test_loss={2:.5f}, test_acc={3:.4%}, num_test_dataset={4}, time={5:.2f}s".format(
            epoch, training_loss / len(train_loader.dataset), test_loss, test_acc, len(test_loader.dataset), time() - ti))
        if test_acc > best_test_acc:  # update best test acc
            best_test_acc = test_acc
            # torch.save(model.state_dict(), os.path.join(args.savepath, '/epoch%d.pkl' % epoch))
            print("Best test_acc increased to {0:.4%}".format(best_test_acc))

    # torch.save(model.state_dict(), os.path.join(args.savepath, 'trained_model.pkl'))
    # print('Trained model saved to \'%s/trained_model.pkl\'' % args.savepath)
    print('\n' + '-'*6 + ' End Training ' + '-'*6)
    print("\n[Total time: {0:.6f} mins = {1:.6f} seconds]".format(
        ((time() - t0) / 60), (time() - t0)))
    return test_pred, test_true

def weights_init(m):
    """
    Model weights initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def load_datasets(path):
    """
    Return peptide datasets.

    :param path: file path of the dataset
    :return: labels(ndarray), t_labels(tensor), t_features(tensor)
    """
    # FILE = "peptide_caps_net.csv"  # 数据集名称
    # DATASETS = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir, "datasets", FILE))  # 构建路径
    # print(os.getcwd())
    DATASETS = os.path.abspath(path)
    print("\nDataset Name:", DATASETS)

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
    t_features = torch.from_numpy(features).contiguous().view(-1, 1, 20, 20).type(torch.FloatTensor)

    return labels, t_onehot, t_features


def load_peptide(features, targets, train_i, test_i, bs=100):
    """
    Construct dataloaders for peptide training and test data for each fold.
    :param features: features tensor
    :param targets: labels tensor
    :param train_i: train dataset index
    :param test_i: test dataset index
    :param bs: batch size of training dataset
    """
    # kwargs = {'num_workers': 0, 'pin_memory': True}
    # Using PyTorch DataLoader
    # 构建 pytorch 数据集
    train_dataset = Data.TensorDataset(
        data_tensor=features[train_i], target_tensor=targets[train_i])
    test_dataset = Data.TensorDataset(data_tensor=features[test_i], target_tensor=targets[test_i])
    # remove num_workers settings
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=bs)
    # test datasets do not use batch process
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=len(test_i))

    return train_loader, test_loader


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
    parser.add_argument("--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=None)
    parser.add_argument('--datapath', required=True, help="Path of dataset.")
    parser.add_argument('--savepath', default='./result',
                        help="The directory for logs and summaries.")
    args = parser.parse_args()
    print("\n", args)
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)

    # define training model
    # peptide 数据集为两分类问题, 样本矩阵大小 20 * 20, 即 data size = [1, 20, 20]
    model = CapsuleNet(input_size=[1, 20, 20], classes=2, routings=3)
    # 输出 CapsNet structure
    print("\n", model)

    # 获取数据集的特征矩阵和分类矩阵
    labels, t_targets, t_features = load_datasets(args.datapath)

    from sklearn.model_selection import KFold
    # 设定 K-fold 分割器
    rs = KFold(n_splits=args.kfolds, shuffle=True, random_state=args.randomseed)
    # 生成 k-fold 训练集、验证集索引
    cv_index_set = rs.split(labels)
    k_fold_step = 1  # 初始化折数
    # 暂存每次选中的测试集和对应预测结果
    test_cache = pred_cache = np.array([], dtype=np.int)

    # Start training and K-fold cross-validation
    for train_index, test_index in cv_index_set:
        # get peptide dataloader
        train_datasets, test_datasets = load_peptide(
            t_features, t_targets, train_index.tolist(), test_index.tolist(), bs=args.batch_size)
        print("\nFold: {0}\n".format(k_fold_step))
        t_pred, t_true = train(model, train_datasets, test_datasets, args)
        # 暂存每次选中的测试集和预测结果
        test_cache = np.concatenate((test_cache, t_true.numpy()))
        pred_cache = np.concatenate((pred_cache, t_pred.numpy()))
        # 每个fold训练结束后次数 +1, Model初始化
        k_fold_step += 1
        model.apply(weights_init)
    
    # k-fold 交叉验证后进行模型评估
    from utils import bi_model_evaluation
    bi_model_evaluation(test_cache, pred_cache)
