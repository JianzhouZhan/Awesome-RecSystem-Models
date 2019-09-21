import re
import os
import math
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from data.Criteo.DCN_dataProcess import EACH_FILE_DATA_NUM
from time import time

EPOCHS = 100
BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DCN_layer(nn.Module):
    def __init__(self, num_dense_feat, num_sparse_feat_list, dropout_deep, deep_layer_sizes,
                 reg_l1=0.01, reg_l2=0.01, num_cross_layers=4):
        super(DCN_layer, self).__init__()
        self.reg_l1 = reg_l1  # L1正则化并没有去使用
        self.reg_l2 = reg_l2
        self.num_dense_feat = num_dense_feat              # denote as D, 连续型特征数量

        # Embedding and Stacking Layer
        embedding_sizes, sparse_feat_embeddings = [], []
        for i, num_sparse_feat in enumerate(num_sparse_feat_list):
            embedding_size = min(num_sparse_feat, 6 * int(np.power(num_sparse_feat, 1/4)))
            embedding_sizes.append(embedding_size)
            feat_embedding = nn.Embedding(num_sparse_feat, embedding_size)
            nn.init.xavier_uniform_(feat_embedding.weight)
            feat_embedding.to(DEVICE)
            sparse_feat_embeddings.append(feat_embedding)
        self.embedding_sizes = embedding_sizes
        self.sparse_feat_embeddings = sparse_feat_embeddings

        self.num_sparse_field = len(num_sparse_feat_list)  # denote as F, 分类型特征field的数量
        self.num_cross_layers = num_cross_layers           # denote as C, Cross层的层数
        self.deep_layer_sizes = deep_layer_sizes           # Deep层中的各神经元的数量

        # Cross Network方面的参数
        self.input_dim = num_dense_feat + sum(embedding_sizes)   # denote as In
        self.cross_bias = nn.Parameter(torch.randn(num_cross_layers, self.input_dim))   # C * In
        nn.init.xavier_uniform_(self.cross_bias)
        self.cross_W = nn.Parameter(torch.randn(num_cross_layers, self.input_dim))
        nn.init.xavier_uniform_(self.cross_W)

        # 神经网络方面的参数
        all_dims = [self.input_dim] + deep_layer_sizes
        for i in range(len(deep_layer_sizes)):
            setattr(self, 'linear_' + str(i + 1), nn.Linear(all_dims[i], all_dims[i + 1]))
            setattr(self, 'batchNorm_' + str(i + 1), nn.BatchNorm1d(all_dims[i + 1]))
            setattr(self, 'dropout_' + str(i + 1), nn.Dropout(dropout_deep[i + 1]))

        # Combination部分: 最后一层全连接层
        self.fc = nn.Linear(self.input_dim + all_dims[-1], 1)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, feat_index_list, dense_x):
        x0 = dense_x
        for i, feat_index in enumerate(feat_index_list):
            sparse_x = self.sparse_feat_embeddings[i](feat_index)                                     # None * F * K
            x0 = torch.cat((x0, sparse_x), dim=1)                                             # None * In

        # Cross Network 部分
        x_cross = x0                                                          # None * In
        for i in range(self.num_cross_layers):
            W = torch.unsqueeze(self.cross_W[i, :].T, dim=1)                  # In * 1
            xT_W = torch.mm(x_cross, W)                                       # None * 1
            x_cross = torch.mul(x0, xT_W) + self.cross_bias[i, :] + x_cross   # None * In

        # Deep Network 部分
        x_deep = x0                                                           # None * In
        for i in range(1, len(self.deep_layer_sizes) + 1):
            x_deep = getattr(self, 'linear_' + str(i))(x_deep)
            x_deep = getattr(self, 'batchNorm_' + str(i))(x_deep)
            x_deep = F.relu(x_deep)
            x_deep = getattr(self, 'dropout_' + str(i))(x_deep)

        x_stack = torch.cat((x_cross, x_deep), dim=1)
        output = self.fc(x_stack)

        return output


""" ************************************************************************************ """
"""                                     训练和测试FM模型                                   """
""" ************************************************************************************ """
def train_DeepFM_model_demo(device):
    """
    训练DeepFM的方式
    :return:
    """
    train_filelist = ["%s%s" % ('../data/Criteo/train_data/', x) for x in os.listdir('../data/Criteo/train_data/')]
    test_filelist = ["%s%s" % ('../data/Criteo/test_data/', x) for x in os.listdir('../data/Criteo/test_data/')]
    train_file_id = [int(re.sub('[\D]', '', x)) for x in train_filelist]
    train_filelist = [train_filelist[idx] for idx in np.argsort(train_file_id)]

    test_file_id = [int(re.sub('[\D]', '', x)) for x in test_filelist]
    test_filelist = [test_filelist[idx] for idx in np.argsort(test_file_id)]

    feat_dict_ = pickle.load(open('../data/Criteo/aid_data/DCN_feat_dict_10.pkl2', 'rb'))
    num_sparse_feat_list = [len(v) + 1 for k, v in feat_dict_.items()]

    # 下面的num_sparse_feat之所以还要加1个维度, 是因为缺失值的处理(详见数据处理过程)
    dcn = DCN_layer(reg_l2=1e-5, num_dense_feat=13, num_sparse_feat_list=num_sparse_feat_list,
                    dropout_deep=[0, 0, 0], deep_layer_sizes=[1024, 1024],
                    num_cross_layers=6).to(DEVICE)
    print("Start Training DeepFM Model!")

    # 定义损失函数还有优化器
    optimizer = torch.optim.Adam(dcn.parameters(), lr=1e-4)

    # 计数train和test的数据量
    train_item_count, test_item_count = 0, 0
    for fname in train_filelist:
        with open(fname.strip(), 'r') as fin:
            for _ in fin:
                train_item_count += 1

    for fname in test_filelist:
        with open(fname.strip(), 'r') as fin:
            for _ in fin:
                test_item_count += 1

    # 由于数据量过大, 如果使用pytorch的DataSet来自定义数据的话, 会耗时很久, 因此, 这里使用其它方式
    for epoch in range(1, EPOCHS + 1):
        tic = time()
        train(dcn, train_filelist, train_item_count, feat_dict_, device, optimizer, epoch)
        toc = time()
        test(dcn, test_filelist, test_item_count, feat_dict_, device)
        print('The Time of Epoch: %.5f min' % float((toc - tic) / 60.0))
        print('The Test Time of Epoch: %.5f min' % float((time() - toc) / 60.0))


def test(model, test_filelist, test_item_count, feat_dict_, device):
    fname_idx = 0
    pred_y, true_y = [], []
    fsparse_features_idxs, dense_features_values, labels = None, None, None
    test_loss = 0
    with torch.no_grad():
        # 不断地取出数据进行计算
        for batch_idx in range(math.ceil(test_item_count / BATCH_SIZE)):
            # 取出当前Batch所在的数据的下标
            st_idx, ed_idx = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE
            ed_idx = min(ed_idx, test_item_count - 1)

            if fsparse_features_idxs is None:
                fsparse_features_idxs, dense_features_values, labels = get_idx_value_label(
                    test_filelist[fname_idx], feat_dict_, shuffle=False)

            st_idx = st_idx - fname_idx * EACH_FILE_DATA_NUM
            ed_idx = ed_idx - fname_idx * EACH_FILE_DATA_NUM

            if ed_idx < EACH_FILE_DATA_NUM:
                batch_fea_idxs = fsparse_features_idxs[st_idx:ed_idx, :]
                batch_fea_values = dense_features_values[st_idx:ed_idx, :]
                batch_labels = labels[st_idx:ed_idx, :]
            else:
                batch_fea_idxs_part1 = fsparse_features_idxs[st_idx:ed_idx, :]
                batch_fea_values_part1 = dense_features_values[st_idx:ed_idx, :]
                batch_labels_part1 = labels[st_idx:ed_idx, :]

                fname_idx += 1
                sparse_features_idxs, dense_features_values, labels = get_idx_value_label(
                    test_filelist[fname_idx], feat_dict_, shuffle=False)
                ed_idx = ed_idx - EACH_FILE_DATA_NUM

                batch_fea_idxs_part2 = fsparse_features_idxs[0:ed_idx, :]
                batch_fea_values_part2 = dense_features_values[0:ed_idx, :]
                batch_labels_part2 = labels[0:ed_idx, :]

                batch_fea_idxs = np.vstack((batch_fea_idxs_part1, batch_fea_idxs_part2))
                batch_fea_values = np.vstack((batch_fea_values_part1, batch_fea_values_part2))
                batch_labels = np.vstack((batch_labels_part1, batch_labels_part2))

            batch_fea_values = torch.from_numpy(batch_fea_values)
            batch_labels = torch.from_numpy(batch_labels)

            sparse_idx_list = []
            for i in range(len(feat_dict_)):
                sparse_idx = batch_fea_idxs[:, i]
                sparse_idx = torch.LongTensor([int(x) for x in sparse_idx])
                sparse_idx = sparse_idx.to(device)
                sparse_idx_list.append(sparse_idx)

            dense_value = batch_fea_values.to(device, dtype=torch.float32)
            target = batch_labels.to(device, dtype=torch.float32)
            output = model(sparse_idx_list, dense_value)

            test_loss += F.binary_cross_entropy_with_logits(output, target)

            pred_y.extend(list(output.cpu().numpy()))
            true_y.extend(list(target.cpu().numpy()))

        print('Roc AUC: %.5f' % roc_auc_score(y_true=np.array(true_y), y_score=np.array(pred_y)))
        test_loss /= math.ceil(test_item_count / BATCH_SIZE)
        print('Test set: Average loss: {:.5f}'.format(test_loss))


def train(model, train_filelist, train_item_count, feat_dict_, device, optimizer, epoch):
    fname_idx = 0
    sparse_features_idxs, dense_features_values, labels = None, None, None
    # 依顺序来遍历访问
    for batch_idx in range(math.ceil(train_item_count / BATCH_SIZE)):
        # 得到当前Batch所在的数据的下标
        st_idx, ed_idx = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE
        ed_idx = min(ed_idx, train_item_count - 1)

        if sparse_features_idxs is None:
            sparse_features_idxs, dense_features_values, labels = get_idx_value_label(
                train_filelist[fname_idx], feat_dict_)

        st_idx = st_idx - fname_idx * EACH_FILE_DATA_NUM
        ed_idx = ed_idx - fname_idx * EACH_FILE_DATA_NUM

        if ed_idx < EACH_FILE_DATA_NUM:
            batch_fea_idxs = sparse_features_idxs[st_idx:ed_idx, :]
            batch_fea_values = dense_features_values[st_idx:ed_idx, :]
            batch_labels = labels[st_idx:ed_idx, :]
        else:
            batch_fea_idxs_part1 = sparse_features_idxs[st_idx:ed_idx, :]
            batch_fea_values_part1 = dense_features_values[st_idx:ed_idx, :]
            batch_labels_part1 = labels[st_idx:ed_idx, :]

            fname_idx += 1
            ed_idx = ed_idx - EACH_FILE_DATA_NUM
            sparse_features_idxs, dense_features_values, labels = get_idx_value_label(
                train_filelist[fname_idx], feat_dict_)

            batch_fea_idxs_part2 = sparse_features_idxs[0:ed_idx, :]
            batch_fea_values_part2 = dense_features_values[0:ed_idx, :]
            batch_labels_part2 = labels[0:ed_idx, :]

            batch_fea_idxs = np.vstack((batch_fea_idxs_part1, batch_fea_idxs_part2))
            batch_fea_values = np.vstack((batch_fea_values_part1, batch_fea_values_part2))
            batch_labels = np.vstack((batch_labels_part1, batch_labels_part2))

        batch_fea_values = torch.from_numpy(batch_fea_values)
        batch_labels = torch.from_numpy(batch_labels)

        sparse_idx_list = []
        for i in range(len(feat_dict_)):
            sparse_idx = batch_fea_idxs[:, i]
            sparse_idx = torch.LongTensor([int(x) for x in sparse_idx])
            sparse_idx = sparse_idx.to(device)
            sparse_idx_list.append(sparse_idx)

        # sparse_idx = torch.LongTensor([[int(x) for x in x_idx] for x_idx in batch_fea_idxs])
        # sparse_idx = sparse_idx.to(device)
        dense_value = batch_fea_values.to(device, dtype=torch.float32)
        target = batch_labels.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        output = model(sparse_idx_list, dense_value)
        loss = F.binary_cross_entropy_with_logits(output, target)

        # regularization_loss = 0
        # for param in model.parameters():
        #     # regularization_loss += model.reg_l1 * torch.sum(torch.abs(param))
        #     regularization_loss += model.reg_l2 * torch.sum(torch.pow(param, 2))
        # loss += regularization_loss

        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{} / {} ({:.0f}%]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(sparse_idx), train_item_count,
                100. * batch_idx / math.ceil(int(train_item_count / BATCH_SIZE)), loss.item()))

def get_idx_value_label(fname, feat_dict_, shuffle=True):
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)

    def _process_line(line):
        features = line.rstrip('\n').split('\t')
        sparse_feat_idx = []
        dense_feat_value = []

        # 对于连续型数据, 根据kaggle Winner的做法, 使用取Log处理
        for idx in continuous_range_:
            if features[idx] == '':
                dense_feat_value.append(0.0)
            else:
                fea_value = math.log(4 + float(features[idx])) if idx == 2 else math.log(1 + float(features[idx]))
                dense_feat_value.append(fea_value)

        # 处理分类型数据, 由于DCN使用Embedding的方式处理, 并不需要value的值, 因此, 仅需要返回Embedding所对应的index即可
        for idx in categorical_range_:
            if features[idx] == '' or features[idx] not in feat_dict_['C' + str(idx)]:
                sparse_feat_idx.append(0)
            else:
                sparse_feat_idx.append(feat_dict_['C' + str(idx)][features[idx]])

        return sparse_feat_idx, dense_feat_value, [int(features[0])]

    sparse_features_idxs, dense_features_values, labels = [], [], []
    with open(fname.strip(), 'r') as fin:
        for line in fin:
            sparse_feat_idx, dense_feat_value, label = _process_line(line)
            sparse_features_idxs.append(sparse_feat_idx)
            dense_features_values.append(dense_feat_value)
            labels.append(label)

    sparse_features_idxs = np.array(sparse_features_idxs)
    dense_features_values = np.array(dense_features_values)
    labels = np.array(labels).astype(np.int32)

    # 进行shuffle
    if shuffle:
        idx_list = np.arange(len(labels))
        np.random.shuffle(idx_list)
        sparse_features_idxs = sparse_features_idxs[idx_list, :]
        dense_features_values = dense_features_values[idx_list, :]
        labels = labels[idx_list, :]
    return sparse_features_idxs, dense_features_values, labels

if __name__ == '__main__':
    train_DeepFM_model_demo(DEVICE)
