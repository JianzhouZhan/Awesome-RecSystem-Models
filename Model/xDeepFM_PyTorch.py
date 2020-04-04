import re
import os
import math
import torch
import numpy as np
import torch.nn as nn
import pickle
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


EPOCHS = 5
BATCH_SIZE = 2048
AID_DATA_DIR = '../data/Criteo/forXDeepFM/'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
PyTorch implementation of Deep & Cross Network[1]

Reference:
[1] xDeepFM: Combining Explicit and Implicit Feature Interactionsfor Recommender Systems,
    Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie,and Guangzhong Sun
    https://arxiv.org/pdf/1803.05170.pdf
[2] TensorFlow implementation of xDeepFM
    https://github.com/Leavingseason/xDeepFM
[3] PaddlePaddle implemantation of xDeepFM
    https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr/xdeepfm
[4] PyTorch implementation of xDeepFM
    https://github.com/qian135/ctr_model_zoo/blob/master/xdeepfm.py
"""


class xDeepFM_layer(nn.Module):
    def __init__(self, num_feat, num_field, dropout_deep, deep_layer_sizes, cin_layer_sizes, split_half=True,
                 reg_l1=0.01, reg_l2=1e-5, embedding_size=10):
        super().__init__()  # Python2 下使用 super(xDeepFM_layer, self).__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.num_feat = num_feat
        self.num_field = num_field
        self.cin_layer_sizes = cin_layer_sizes
        self.deep_layer_sizes = deep_layer_sizes
        self.embedding_size = embedding_size    # denoted by M
        self.dropout_deep = dropout_deep
        self.split_half = split_half

        self.input_dim = num_field * embedding_size

        # init feature embedding
        feat_embedding = nn.Embedding(num_feat, embedding_size)
        nn.init.xavier_uniform_(feat_embedding.weight)
        self.feat_embedding = feat_embedding

        # Compress Interaction Network (CIN) Part
        cin_layer_dims = [self.num_field] + cin_layer_sizes

        prev_dim, fc_input_dim = self.num_field, 0
        self.conv1ds = nn.ModuleList()
        for k in range(1, len(cin_layer_dims)):
            conv1d = nn.Conv1d(cin_layer_dims[0] * prev_dim, cin_layer_dims[k], 1)
            nn.init.xavier_uniform_(conv1d.weight)
            self.conv1ds.append(conv1d)
            if self.split_half and k != len(self.cin_layer_sizes):
                prev_dim = cin_layer_dims[k] // 2
            else:
                prev_dim = cin_layer_dims[k]
            fc_input_dim += prev_dim

        # Deep Neural Network Part
        all_dims = [self.input_dim] + deep_layer_sizes
        for i in range(len(deep_layer_sizes)):
            setattr(self, 'linear_' + str(i + 1), nn.Linear(all_dims[i], all_dims[i + 1]))
            setattr(self, 'batchNorm_' + str(i + 1), nn.BatchNorm1d(all_dims[i + 1]))
            setattr(self, 'dropout_' + str(i + 1), nn.Dropout(dropout_deep[i + 1]))

        # Linear Part
        self.linear = nn.Linear(self.input_dim, 1)

        # output Part
        self.output_layer = nn.Linear(1 + fc_input_dim + deep_layer_sizes[-1], 1)

    def forward(self, feat_index, feat_value, use_dropout=True):
        # get feat embedding
        fea_embedding = self.feat_embedding(feat_index)    # None * F * K
        x0 = fea_embedding

        # Linear Part
        linear_part = self.linear(fea_embedding.reshape(-1, self.input_dim))

        # CIN Part
        x_list = [x0]
        res = []
        for k in range(1, len(self.cin_layer_sizes) + 1):
            # Batch * H_K * D, Batch * M * D -->  Batch * H_k * M * D
            z_k = torch.einsum('bhd,bmd->bhmd', x_list[-1], x_list[0])
            z_k = z_k.reshape(x0.shape[0], x_list[-1].shape[1] * x0.shape[1], x0.shape[2])
            x_k = self.conv1ds[k - 1](z_k)
            x_k = torch.relu(x_k)

            if self.split_half and k != len(self.cin_layer_sizes):
                # x, h = torch.split(x, x.shape[1] // 2, dim=1)
                next_hidden, hi = torch.split(x_k, x_k.shape[1] // 2, 1)
            else:
                next_hidden, hi = x_k, x_k

            x_list.append(next_hidden)
            res.append(hi)

        res = torch.cat(res, dim=1)
        res = torch.sum(res, dim=2)

        # Deep NN Part
        y_deep = fea_embedding.reshape(-1, self.num_field * self.embedding_size)  # None * (F * K)
        if use_dropout:
            y_deep = nn.Dropout(self.dropout_deep[0])(y_deep)

        for i in range(1, len(self.deep_layer_sizes) + 1):
            y_deep = getattr(self, 'linear_' + str(i))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(i))(y_deep)
            y_deep = F.relu(y_deep)
            if use_dropout:
                y_deep = getattr(self, 'dropout_' + str(i))(y_deep)

        # Output Part
        concat_input = torch.cat((linear_part, res, y_deep), dim=1)
        output = self.output_layer(concat_input)
        return output


""" ************************************************************************************ """
"""                                   训练和测试xDeepFM模型                                """
""" ************************************************************************************ """
def train_xDeepFM_model_demo(device):
    """
    训练DeepFM的方式
    :return:
    """
    train_filelist = ["%s%s" % (AID_DATA_DIR + 'train_data/', x) for x in os.listdir(AID_DATA_DIR + 'train_data/')]
    test_filelist = ["%s%s" % (AID_DATA_DIR + 'test_data/', x) for x in os.listdir(AID_DATA_DIR + 'test_data/')]
    train_file_id = [int(re.sub('[\D]', '', x)) for x in train_filelist]
    train_filelist = [train_filelist[idx] for idx in np.argsort(train_file_id)]

    test_file_id = [int(re.sub('[\D]', '', x)) for x in test_filelist]
    test_filelist = [test_filelist[idx] for idx in np.argsort(test_file_id)]

    featIndex = pickle.load(open(AID_DATA_DIR + 'aid_data/feat_dict_4.pkl2', 'rb'))
    feat_cnt = pickle.load(open(AID_DATA_DIR + 'aid_data/feat_cnt_4.pkl2', 'rb'))

    # 下面的num_feat的长度还需要考虑缺失值的处理而多了一个维度
    xdeepfm = xDeepFM_layer(num_feat=len(featIndex) + 1, num_field=39, dropout_deep=[0, 0, 0, 0, 0],
                            deep_layer_sizes=[400, 400, 400, 400], cin_layer_sizes=[100, 100, 50],
                            embedding_size=16).to(DEVICE)
    print("Start Training DeepFM Model!")

    # 定义损失函数还有优化器
    optimizer = torch.optim.Adam(xdeepfm.parameters())

    # 计数train和test的数据量
    train_item_count = get_in_filelist_item_num(train_filelist)
    test_item_count = get_in_filelist_item_num(test_filelist)

    # 由于数据量过大, 如果使用pytorch的DataSet来自定义数据的话, 会耗时很久, 因此, 这里使用其它方式
    for epoch in range(1, EPOCHS + 1):
        train(xdeepfm, train_filelist, train_item_count, featIndex, feat_cnt, device, optimizer, epoch)
        test(xdeepfm, test_filelist, test_item_count, featIndex, feat_cnt, device)


def get_in_filelist_item_num(filelist):
    count = 0
    for fname in filelist:
        with open(fname.strip(), 'r') as fin:
            for _ in fin:
                count += 1
    return count


def test(model, test_filelist, test_item_count, featIndex, feat_cnt, device):
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
                    test_filelist[fname_idx], featIndex, feat_cnt, shuffle=False)

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
                    test_filelist[fname_idx], featIndex, feat_cnt, shuffle=False)
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


def train(model, train_filelist, train_item_count, featIndex, feat_cnt, device, optimizer, epoch, use_reg_l2=True):
    fname_idx = 0
    features_idxs, features_values, labels = None, None, None

    # 依顺序来遍历访问
    pre_file_data_count = 0  # 记录在前面已经访问的文件中的数据的数量
    for batch_idx in range(math.ceil(train_item_count / BATCH_SIZE)):
        # 得到当前Batch所要取的数据的起始及终止下标
        st_idx, ed_idx = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE
        ed_idx = min(ed_idx, train_item_count - 1)

        if features_idxs is None:
            features_idxs, features_values, labels = get_idx_value_label(train_filelist[fname_idx], featIndex, feat_cnt)

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
            features_idxs, features_values, labels = get_idx_value_label(train_filelist[fname_idx], featIndex, feat_cnt)
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

        if use_reg_l2:
            regularization_loss = 0
            for param in model.parameters():
                # regularization_loss += model.reg_l1 * torch.sum(torch.abs(param))
                regularization_loss += model.reg_l2 * torch.sum(torch.pow(param, 2))
            loss += regularization_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{} / {} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(idx), train_item_count,
                100. * batch_idx / math.ceil(int(train_item_count / BATCH_SIZE)), loss.item()))


def get_idx_value_label(fname, featIndex, feat_cnt, shuffle=True):
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)

    def _process_line(line):
        features = line.rstrip('\n').split('\t')
        feat_idx = []
        feat_value = []

        for idx in continuous_range_:
            key = 'I' + str(idx)
            val = features[idx]

            if val == '':
                feat = str(key) + '#' + 'absence'
            else:
                val = int(float(val))
                if val > 2:
                    val = int(math.log(float(val)) ** 2)
                else:
                    val = 'SP' + str(val)
                feat = str(key) + '#' + str(val)

            feat_idx.append(featIndex[feat])
            feat_value.append(1)

        for idx in categorical_range_:
            key = 'C' + str(idx - 13)
            val = features[idx]

            if val == '':
                feat = str(key) + '#' + 'absence'
            else:
                feat = str(key) + '#' + str(val)
            if feat_cnt[feat] > 4:
                feat = feat
            else:
                feat = str(key) + '#' + str(feat_cnt[feat])

            feat_idx.append(featIndex[feat])
            feat_value.append(1)
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
    train_xDeepFM_model_demo(DEVICE)
