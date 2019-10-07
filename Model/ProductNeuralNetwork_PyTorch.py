import re
import os
import math
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

EPOCHS = 10
BATCH_SIZE = 2048
AID_DATA_DIR = '../data/Criteo/forPNN/'  # 辅助用途的文件路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
PyTorch implementation of Product-based Neural Network[1]

Reference:
[1] Product-based Neural Networks for User ResponsePrediction,
    Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, Jun Wang
[2] Tensorflow implementation of PNN
    https://github.com/Atomu2014/product-nets
"""

class PNN_layer(nn.Module):

    def __init__(self, num_feat, num_field, dropout_deep, deep_layer_sizes, product_layer_dim=10,
                 reg_l1=0.01, reg_l2=1e-5, embedding_size=10, product_type='outer'):
        super(PNN_layer, self).__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.num_feat = num_feat                                   # Denoted as
        self.num_field = num_field                                 # Denoted as N
        self.product_layer_dim = product_layer_dim                 # Denoted as D1
        self.dropout_deep = dropout_deep

        # Embedding
        feat_embeddings = nn.Embedding(num_feat, embedding_size)
        nn.init.xavier_uniform_(feat_embeddings.weight)
        self.feat_embeddings = feat_embeddings

        # linear part
        linear_weights = torch.randn((product_layer_dim, num_field, embedding_size))   # D1 * N * M
        nn.init.xavier_uniform_(linear_weights)
        self.linear_weights = nn.Parameter(linear_weights)

        # quadratic part
        self.product_type = product_type
        if product_type == 'inner':
            theta = torch.randn((product_layer_dim, num_field))        # D1 * N
            nn.init.xavier_uniform_(theta)
            self.theta = nn.Parameter(theta)
        else:
            quadratic_weights = torch.randn((product_layer_dim, embedding_size, embedding_size))  # D1 * M * M
            nn.init.xavier_uniform_(quadratic_weights)
            self.quadratic_weights = nn.Parameter(quadratic_weights)

        # fc layer
        self.deep_layer_sizes = deep_layer_sizes
        all_dims = [self.product_layer_dim + self.product_layer_dim] + deep_layer_sizes
        for i in range(1, len(deep_layer_sizes) + 1):
            setattr(self, 'linear_' + str(i), nn.Linear(all_dims[i - 1], all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout_deep[i]))

        # last layer
        self.fc = nn.Linear(deep_layer_sizes[-1], 1)

    def forward(self, feat_index, feat_value):
        # embedding part
        feat_embedding = self.feat_embeddings(feat_index)          # Batch * N * M

        # linear part
        lz = torch.einsum('bnm,dnm->bd', feat_embedding, self.linear_weights)  # Batch * D1

        # quadratic part
        if self.product_type == 'inner':
            theta = torch.einsum('bnm,dn->bdnm', feat_embedding, self.theta)            # Batch * D1 * N * M
            lp = torch.einsum('bdnm,bdnm->bd', theta, theta)
        else:
            embed_sum = torch.sum(feat_embedding, dim=1)
            p = torch.einsum('bm,bm->bmm', embed_sum, embed_sum)
            lp = torch.einsum('bmm,dmm->bd', p, self.quadratic_weights)        # Batch * D1

        y_deep = torch.cat((lz, lp), dim=1)
        y_deep = nn.Dropout(self.dropout_deep[0])(y_deep)

        for i in range(1, len(self.deep_layer_sizes) + 1):
            y_deep = getattr(self, 'linear_' + str(i))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(i))(y_deep)
            y_deep = F.relu(y_deep)
            y_deep = getattr(self, 'dropout_' + str(i))(y_deep)

        output = self.fc(y_deep)
        return output


""" ************************************************************************************ """
"""                                     训练和测试PNN模型                                  """
""" ************************************************************************************ """
def train_PNN_model_demo(device):
    """
    训练PNN的方式
    :return:
    """
    train_filelist = ["%s%s" % (AID_DATA_DIR + 'train_data/', x) for x in os.listdir(AID_DATA_DIR + 'train_data/')]
    test_filelist = ["%s%s" % (AID_DATA_DIR + 'test_data/', x) for x in os.listdir(AID_DATA_DIR + 'test_data/')]
    train_file_id = [int(re.sub('[\D]', '', x)) for x in train_filelist]
    train_filelist = [train_filelist[idx] for idx in np.argsort(train_file_id)]

    test_file_id = [int(re.sub('[\D]', '', x)) for x in test_filelist]
    test_filelist = [test_filelist[idx] for idx in np.argsort(test_file_id)]

    feat_dict_ = pickle.load(open(AID_DATA_DIR + 'aid_data/feat_dict_10.pkl2', 'rb'))

    # 下面的num_feat的长度还需要考虑缺失值的处理而多了一个维度
    deepfm = PNN_layer(num_feat=len(feat_dict_) + 1, num_field=39, dropout_deep=[0.5, 0.5, 0.5],
                       deep_layer_sizes=[400, 400], product_layer_dim=10,
                       reg_l1=0.01, reg_l2=1e-5, embedding_size=10, product_type='inner').to(DEVICE)
    print("Start Training PNN Model!")

    # 定义损失函数还有优化器
    optimizer = torch.optim.Adam(deepfm.parameters())

    # 计数train和test的数据量
    train_item_count = get_in_filelist_item_num(train_filelist)
    test_item_count = get_in_filelist_item_num(test_filelist)

    # 由于数据量过大, 如果使用pytorch的DataSet来自定义数据集的话, 会耗时很久, 因此, 这里使用其它方式
    for epoch in range(1, EPOCHS + 1):
        train(deepfm, train_filelist, train_item_count, feat_dict_, device, optimizer, epoch)
        test(deepfm, test_filelist, test_item_count, feat_dict_, device)


def get_in_filelist_item_num(filelist):
    count = 0
    for fname in filelist:
        with open(fname.strip(), 'r') as fin:
            for _ in fin:
                count += 1
    return count


def test(model, test_filelist, test_item_count, feat_dict_, device):
    fname_idx = 0
    pred_y, true_y = [], []
    features_idxs, features_values, labels = None, None, None
    test_loss = 0
    with torch.no_grad():
        # 不断地取出数据进行计算
        pre_file_data_count = 0  # 记录在前面已经访问的文件中的数据的数量
        for batch_idx in range(math.ceil(test_item_count / BATCH_SIZE)):
            # 取出当前Batch所在的数据的下标
            st_idx, ed_idx = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE
            ed_idx = min(ed_idx, test_item_count - 1)

            if features_idxs is None:
                features_idxs, features_values, labels = get_idx_value_label(
                    test_filelist[fname_idx], feat_dict_, shuffle=False)

            # 得到在现有文件中的所对应的起始位置及终止位置
            st_idx -= pre_file_data_count
            ed_idx -= pre_file_data_count

            # 如果数据越过当前文件所对应的范围时, 则再读取下一个文件
            if ed_idx <= len(features_idxs):
                batch_fea_idxs = features_idxs[st_idx:ed_idx, :]
                batch_fea_values = features_values[st_idx:ed_idx, :]
                batch_labels = labels[st_idx:ed_idx, :]
            else:
                pre_file_data_count += len(features_idxs)

                # 得到在这个文件内的数据
                batch_fea_idxs_part1 = features_idxs[st_idx::, :]
                batch_fea_values_part1 = features_values[st_idx::, :]
                batch_labels_part1 = labels[st_idx::, :]

                # 得到在下一个文件内的数据
                fname_idx += 1
                ed_idx -= len(features_idxs)
                features_idxs, features_values, labels = get_idx_value_label(
                    test_filelist[fname_idx], feat_dict_, shuffle=False)
                batch_fea_idxs_part2 = features_idxs[0:ed_idx, :]
                batch_fea_values_part2 = features_values[0:ed_idx, :]
                batch_labels_part2 = labels[0:ed_idx, :]

                # 将两部分数据进行合并(正常情况下, 数据最多只会在两个文件中)
                batch_fea_idxs = np.vstack((batch_fea_idxs_part1, batch_fea_idxs_part2))
                batch_fea_values = np.vstack((batch_fea_values_part1, batch_fea_values_part2))
                batch_labels = np.vstack((batch_labels_part1, batch_labels_part2))

            # 进行格式转换
            batch_fea_values = torch.from_numpy(batch_fea_values)
            batch_labels = torch.from_numpy(batch_labels)

            idx = torch.LongTensor([[int(x) for x in x_idx] for x_idx in batch_fea_idxs])
            idx = idx.to(device)
            value = batch_fea_values.to(device, dtype=torch.float32)
            target = batch_labels.to(device, dtype=torch.float32)
            output = model(idx, value)

            test_loss += F.binary_cross_entropy_with_logits(output, target)

            pred_y.extend(list(output.cpu().numpy()))
            true_y.extend(list(target.cpu().numpy()))

        print('Roc AUC: %.5f' % roc_auc_score(y_true=np.array(true_y), y_score=np.array(pred_y)))
        test_loss /= math.ceil(test_item_count / BATCH_SIZE)
        print('Test set: Average loss: {:.5f}'.format(test_loss))


def train(model, train_filelist, train_item_count, feat_dict_, device, optimizer, epoch):
    fname_idx = 0
    features_idxs, features_values, labels = None, None, None

    # 依顺序来遍历访问
    pre_file_data_count = 0  # 记录在前面已经访问的文件中的数据的数量
    for batch_idx in range(math.ceil(train_item_count / BATCH_SIZE)):
        # 得到当前Batch所要取的数据的起始及终止下标
        st_idx, ed_idx = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE
        ed_idx = min(ed_idx, train_item_count - 1)

        if features_idxs is None:
            features_idxs, features_values, labels = get_idx_value_label(train_filelist[fname_idx], feat_dict_)

        # 得到在现有文件中的所对应的起始位置及终止位置
        st_idx -= pre_file_data_count
        ed_idx -= pre_file_data_count

        # 如果数据越过当前文件所对应的范围时, 则再读取下一个文件
        if ed_idx < len(features_idxs):
            batch_fea_idxs = features_idxs[st_idx:ed_idx, :]
            batch_fea_values = features_values[st_idx:ed_idx, :]
            batch_labels = labels[st_idx:ed_idx, :]
        else:
            pre_file_data_count += len(features_idxs)

            # 得到在这个文件内的数据
            batch_fea_idxs_part1 = features_idxs[st_idx::, :]
            batch_fea_values_part1 = features_values[st_idx::, :]
            batch_labels_part1 = labels[st_idx::, :]

            # 得到在下一个文件内的数据
            fname_idx += 1
            ed_idx -= len(features_idxs)
            features_idxs, features_values, labels = get_idx_value_label(train_filelist[fname_idx], feat_dict_)
            batch_fea_idxs_part2 = features_idxs[0:ed_idx, :]
            batch_fea_values_part2 = features_values[0:ed_idx, :]
            batch_labels_part2 = labels[0:ed_idx, :]

            # 将两部分数据进行合并(正常情况下, 数据最多只会在两个文件中)
            batch_fea_idxs = np.vstack((batch_fea_idxs_part1, batch_fea_idxs_part2))
            batch_fea_values = np.vstack((batch_fea_values_part1, batch_fea_values_part2))
            batch_labels = np.vstack((batch_labels_part1, batch_labels_part2))

        # 进行格式转换
        batch_fea_values = torch.from_numpy(batch_fea_values)
        batch_labels = torch.from_numpy(batch_labels)

        idx = torch.LongTensor([[int(x) for x in x_idx] for x_idx in batch_fea_idxs])
        idx = idx.to(device)
        value = batch_fea_values.to(device, dtype=torch.float32)
        target = batch_labels.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        output = model(idx, value)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{} / {} ({:.0f}%]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(idx), train_item_count,
                100. * batch_idx / math.ceil(int(train_item_count / BATCH_SIZE)), loss.item()))


def get_idx_value_label(fname, feat_dict_, shuffle=True):
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)
    cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cont_max_ = [5775, 257675, 65535, 969, 23159456, 431037, 56311, 6047, 29019, 46, 231, 4008, 7393]
    cont_diff_ = [cont_max_[i] - cont_min_[i] for i in range(len(cont_min_))]

    def _process_line(line):
        features = line.rstrip('\n').split('\t')
        feat_idx = []
        feat_value = []

        # MinMax标准化连续型数据
        for idx in continuous_range_:
            if features[idx] == '':
                feat_idx.append(0)
                feat_value.append(0.0)
            else:
                feat_idx.append(feat_dict_[idx])
                feat_value.append((float(features[idx]) - cont_min_[idx - 1]) / cont_diff_[idx - 1])

        # 处理分类型数据
        for idx in categorical_range_:
            if features[idx] == '' or features[idx] not in feat_dict_:
                feat_idx.append(0)
                feat_value.append(0.0)
            else:
                feat_idx.append(feat_dict_[features[idx]])
                feat_value.append(1.0)

        return feat_idx, feat_value, [int(features[0])]

    features_idxs, features_values, labels = [], [], []
    with open(fname.strip(), 'r') as fin:
        for line in fin:
            feat_idx, feat_value, label = _process_line(line)
            features_idxs.append(feat_idx)
            features_values.append(feat_value)
            labels.append(label)

    features_idxs = np.array(features_idxs)
    features_values = np.array(features_values)
    labels = np.array(labels).astype(np.int32)

    # 进行shuffle
    if shuffle:
        idx_list = np.arange(len(features_idxs))
        np.random.shuffle(idx_list)

        features_idxs = features_idxs[idx_list, :]
        features_values = features_values[idx_list, :]
        labels = labels[idx_list, :]
    return features_idxs, features_values, labels

if __name__ == '__main__':
    train_PNN_model_demo(DEVICE)



