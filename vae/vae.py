# -*- coding: utf-8 -*-
"""
A PyTorch implementation of VAE
Modified: liumin@shmtu.edu.cn
Reference: `https://github.com/pytorch/examples/tree/master/vae`
"""
from __future__ import division, print_function
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from torch import nn, optim
from torch.nn import functional as F


class VAE(nn.Module):
    """
    VAE Model

    Args:
        input_size: Input size of original feature vector
        h_dim: Number of hidden layer units
        z_dim: Length of encoded feature vector 

    """

    def __init__(self, input_size, h_dim, z_dim):
        super(VAE, self).__init__()
        self.x_dim = input_size
        self.fc1 = nn.Linear(input_size, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, input_size):
    """
    Loss Function of VAE
    
    Args:
        recon_x: generating data
        x: origin data
        mu: latent mean
        logvar: latent log variance
        input_size: length of feature vector
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_size), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch, model, train_loader, input_size):
    """
    Training a VAE

    Args:
        epoch: Number of training epochs
        model: Initialized VAE model
        train_loader: torch.utils.data.DataLoader for training data
        input_size: Input size of original feature vector
    """
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, input_size)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # if batch_idx % 2 == 0:
        #     # print('\nBatch_X:', data.numpy(), data.shape)
        #     print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


# New DataSet access method in PyTorch 0.40 (Custom DataSet)
class CustomDataset(Data.Dataset):
    """
    Custom dataset, wrapping data and target tensors

    Refs: torch.utils.data.TensorDataset

    All other datasets should subclass ``~torch.utils.data.Dataset``. 
    All subclasses should override ``__len__``, that provides the size of the dataset, and ``__getitem__``, 
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the dataset csv file with annotations.
        """
        DATASETS = os.path.abspath(csv_file)
        df = pd.read_csv(DATASETS)
        # 第1列为分类标注名称
        labels = df.values[:, 0].astype(np.int32)
        # 调整 label 2为1(N样本), 1为0(P样本), 便于转换为 One-hot Vector
        labels[labels == 1] = 0
        labels[labels == 2] = 1
        t_labels = torch.from_numpy(labels).view(-1, 1)
        # 转换为 One-hot Vector
        self.target_tensor = torch.zeros(labels.size, 2)
        self.target_tensor.scatter_(1, t_labels.type(torch.LongTensor), 1)
        # 第2列之后为特征矩阵
        features = df.values[:, 1:].astype(np.float)
        # 特征矩阵数值缩放 (0, 1) 区间
        from sklearn.preprocessing import MinMaxScaler
        min_max_scaler = MinMaxScaler()
        features_minmax = min_max_scaler.fit_transform(features)
        self.data_tensor = torch.from_numpy(features_minmax).type(torch.FloatTensor)

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def get_len_feature(self):
        return self.data_tensor.size(1)

    def get_len_samples(self):
        # 负样本个数
        len_n = torch.argmax(self.target_tensor, dim=1).sum()
        # 正样本个数
        len_p = self.target_tensor.size(0) - len_n
        return len_p.numpy(), len_n.numpy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='VAE for citrullination')
    parser.add_argument('-b', '--batchsize', type=int, default=256, metavar='N',
                        help='input batch size for training.')
    parser.add_argument('-u', '--nunits', type=int,
                        help="number of hidden layer units", default=512)
    parser.add_argument('-o', '--encode', type=int,
                        help="Length of encoded feature-vector", default=128)
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--datapath', required=True, help="Path of dataset")
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')

    # 参数初始化
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # 输出参数信息
    print("\n", args)
    # 手动设定随机种子，确保每次测试数据分布一致
    torch.manual_seed(args.seed)
    # 是否应用 GPU 加速
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # 获取数据集的特征矩阵和分类矩阵
    # t_targets, t_features, len_features = load_datasets(args.datapath)
    # 构造数据集
    t_dataset = CustomDataset(args.datapath)
    # 初始化 VAE 训练模型
    vae_model = VAE(t_dataset.get_len_feature(), args.nunits, args.encode).to(device)
    print("\n", vae_model)
    # 设定优化器
    optimizer = optim.Adam(vae_model.parameters(), lr=1e-2)

    # 构建 PyTorch DataLoader, VAE 不需要测试集和标注集
    train_loader = Data.DataLoader(dataset=t_dataset, batch_size=args.batchsize, shuffle=True)

    # for epoch in range(1, args.epochs + 1):
    #     train(epoch, vae_model, train_loader, args.encode)

    l_feature = t_dataset.get_len_feature()
    n_p, n_n = t_dataset.get_len_samples()
    print("\nLength of feature vector:", l_feature)
    print("\nNumber of positive samples:", n_p, ", Number of negative samples:", n_n)
    # print("\nFeatures Matrix:", t_dataset.data_tensor)

    for epoch in range(1, args.epochs + 1):
        train(epoch, vae_model, train_loader, l_feature)
    
    # Output encoded features matrix
    encoded_martrix, _ = vae_model.encode(t_dataset.data_tensor)
    # print(encoded_martrix)
