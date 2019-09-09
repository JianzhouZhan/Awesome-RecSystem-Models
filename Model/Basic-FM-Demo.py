import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from sklearn import preprocessing

EPOCHS = 50
BATCH_SIZE = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


""" ************************************************************************************ """
"""                                      数据读取和转换                                    """
""" ************************************************************************************ """
def load_dataset():
    header = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    df_user = pd.read_csv('../data/FM-Data/u.user', sep='|', names=header)
    header = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
              'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    df_item = pd.read_csv('../data/FM-Data/u.item', sep='|', names=header, encoding="ISO-8859-1")
    df_item = df_item.drop(columns=['title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown'])

    df_user['age'] = pd.cut(df_user['age'], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                            labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90',
                                    '90-100'])
    df_user = pd.get_dummies(df_user, columns=['gender', 'occupation', 'age'])
    df_user = df_user.drop(columns=['zip_code'])

    user_features = df_user.columns.values.tolist()
    movie_features = df_item.columns.values.tolist()
    cols = user_features + movie_features

    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df_train = pd.read_csv('../data/FM-Data/ua.base', sep='\t', names=header)
    # df_train['rating'] = df_train.rating.apply(lambda x: 1 if int(x) == 4 else 0)
    df_train = df_train.merge(df_user, on='user_id', how='left')
    df_train = df_train.merge(df_item, on='item_id', how='left')

    df_test = pd.read_csv('../data/FM-Data/ua.test', sep='\t', names=header)
    # df_test['rating'] = df_test.rating.apply(lambda x: 1 if int(x) == 4 else 0)
    df_test = df_test.merge(df_user, on='user_id', how='left')
    df_test = df_test.merge(df_item, on='item_id', how='left')
    train_labels = np.array(df_train['rating'].astype(np.int32))
    test_labels = np.array(df_test['rating'].astype(np.int32))
    return df_train[cols].values, train_labels, df_test[cols].values, test_labels


class FM_layer(nn.Module):
    def __init__(self, class_num=1, feature_num=10, latent_factor_dim=5):
        super(FM_layer, self).__init__()
        self.fea_num = feature_num
        self.k = latent_factor_dim
        self.linear = nn.Linear(self.fea_num, class_num)  # 前两项的线性层
        self.v = nn.Parameter(torch.randn(self.fea_num, self.k, class_num))  # 交互矩阵

    def forward(self, x):
        # 先计算得到线性的那一部分
        linear_part = self.linear(x)

        # 再计算得到交互的那一个部分
        # 为了使用矩阵计算, (Batch * fea_num) * (fea_num * k * class_num), 我们使用Tensor的转置来处理
        print(self.v)
        print(x)
        interaction_part1 = torch.matmul(self.v.permute(2, 1, 0), x.T).permute(2, 1, 0)
        print(interaction_part1)
        interaction_part1 = torch.pow(interaction_part1, 2)
        interaction_part1 = -0.5 * torch.sum(interaction_part1, dim=1)
        interaction_part1 = torch.squeeze(interaction_part1, dim=1)

        x_square, v_square = torch.pow(x, 2), torch.pow(self.v, 2)
        interaction_part2 = torch.matmul(v_square.permute(2, 1, 0), x_square.T).permute(2, 1, 0)
        interaction_part2 = 0.5 * torch.sum(interaction_part2, dim=1)
        interaction_part2 = torch.squeeze(interaction_part2, dim=1)
        output = linear_part + interaction_part1 + interaction_part2

        output = F.softmax(output, dim=1)
        return output


""" ************************************************************************************ """
"""                                      数据读取和装换                                    """
""" ************************************************************************************ """
def train_FM_model_demo():

    # Step1: 导入数据
    x_train, y_train, x_test, y_test = load_dataset()
    x_train = preprocessing.scale(x_train, with_mean=True, with_std=True)
    x_test = preprocessing.scale(x_test, with_mean=True, with_std=True)
    class_num = len(set([y for y in y_train] + [y for y in y_test]))

    # FM模型
    fm = FM_layer(class_num=class_num, feature_num=x_train.shape[1], latent_factor_dim=40).to(DEVICE)

    # 定义损失函数还有优化器
    optm = torch.optim.Adam(fm.parameters())

    train_loader = get_batch_loader(x_train, y_train, shuffle=True)
    test_loader = get_batch_loader(x_test, y_test, shuffle=False)

    for epoch in range(1, EPOCHS + 1):
        train(fm, DEVICE, train_loader, optm, epoch)
        test(fm, DEVICE, test_loader)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float32), target.to(device).long()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 500 == 0:
            print('Train Epoch: {} [{} / {} ({:.0f}%]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float32), target.to(device).long()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))


def get_batch_loader(features, labels, shuffle=True):
    class MyDataset(data.Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __getitem__(self, index):  # 返回的是tensor
            img, target = self.features[index], self.labels[index]
            return img, target

        def __len__(self):
            return len(self.features)

    train_loader = data.DataLoader(MyDataset(features, labels), batch_size=BATCH_SIZE, shuffle=shuffle)
    return train_loader


if __name__ == '__main__':
    train_FM_model_demo()
