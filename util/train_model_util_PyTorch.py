import re
import os
import math
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

EPOCHS = 5
BATCH_SIZE = 2048


# def train_model_v2(model, device, train_label_path, train_idx_path, train_value_path, optimizer, epoch,
#                    use_reg_l1=False, use_reg_l2=False):
#     train_label, train_idx, train_value = [], [], []
#     with open(train_label_path.strip(), 'r') as fin:
#         for line in fin:
#             train_label.append(line.rstrip('\n'))
#     print("*******************************************")
#     with open(train_idx_path.strip(), 'r') as fin:
#         for line in fin:
#             train_idx.append(line.rstrip('\n'))
#     print("*******************************************")
#     with open(train_value_path.strip(), 'r') as fin:
#         for line in fin:
#             train_value.append(line.rstrip('\n'))
#
#     print("*******************************************")
#
#     train_item_count = len(train_label)
#     for batch_idx in range(math.ceil(train_item_count / BATCH_SIZE)):
#         # 得到当前Batch所要取的数据的起始及终止下标
#         st_idx, ed_idx = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE
#         ed_idx = min(ed_idx, train_item_count - 1)
#
#         batch_fea_idxs = train_idx[st_idx:ed_idx, :]
#         batch_fea_values = train_value[st_idx:ed_idx, :]
#         batch_labels = train_label[st_idx:ed_idx, :]
#
#         batch_fea_idxs = np.array([int(x) for x in batch_fea_idxs])
#         batch_fea_values = np.array(np.array(re.split(x, '\t'), dtype=float) for x in batch_fea_values)
#         batch_labels = np.array(np.array(re.split(x, '\t'), dtype=int) for x in batch_labels)
#
#         batch_fea_values = torch.from_numpy(batch_fea_values)
#         batch_labels = torch.from_numpy(batch_labels)
#
#         idx = torch.LongTensor([[int(x) for x in x_idx] for x_idx in batch_fea_idxs])
#         idx = idx.to(device)
#         value = batch_fea_values.to(device, dtype=torch.float32)
#         target = batch_labels.to(device, dtype=torch.float32)
#         optimizer.zero_grad()
#         output = model(idx, value)
#         loss = F.binary_cross_entropy_with_logits(output, target)
#
#         if use_reg_l1:
#             for param in model.parameters():
#                 loss += model.reg_l1 * torch.sum(torch.abs(param))
#         if use_reg_l2:
#             for param in model.parameters():
#                 loss += model.reg_l2 * torch.sum(torch.pow(param, 2))
#
#         loss.backward()
#
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
#         optimizer.step()
#         if batch_idx % 1000 == 0:
#             print('Train Epoch: {} [{} / {} ({:.0f}%)]\tLoss:{:.6f}'.format(
#                 epoch, batch_idx * len(idx), train_item_count,
#                        100. * batch_idx / math.ceil(int(train_item_count / BATCH_SIZE)), loss.item()))
#
#
# def test_model_v2(model, device, train_label_path, train_idx_path, train_value_path):
#     train_label, train_idx, train_value = [], [], []
#     with open(train_label_path.strip(), 'r') as fin:
#         for line in fin:
#             train_label.append(line.rstrip('\n'))
#     with open(train_idx_path.strip(), 'r') as fin:
#         for line in fin:
#             train_idx.append(line.rstrip('\n'))
#     with open(train_value_path.strip(), 'r') as fin:
#         for line in fin:
#             train_value.append(line.rstrip('\n'))
#
#     test_item_count = len(train_label)
#     test_loss = 0
#     pred_y, true_y = [], []
#
#     with torch.no_grad():
#         train_item_count = len(train_label)
#         for batch_idx in range(math.ceil(train_item_count / BATCH_SIZE)):
#             # 得到当前Batch所要取的数据的起始及终止下标
#             st_idx, ed_idx = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE
#             ed_idx = min(ed_idx, train_item_count - 1)
#
#             batch_fea_idxs = train_idx[st_idx:ed_idx, :]
#             batch_fea_values = train_value[st_idx:ed_idx, :]
#             batch_labels = train_label[st_idx:ed_idx, :]
#
#             batch_fea_idxs = np.array([int(x) for x in batch_fea_idxs])
#             batch_fea_values = np.array(np.array(re.split(x, '\t'), dtype=float) for x in batch_fea_values)
#             batch_labels = np.array(np.array(re.split(x, '\t'), dtype=int) for x in batch_labels)
#
#             batch_fea_values = torch.from_numpy(batch_fea_values)
#             batch_labels = torch.from_numpy(batch_labels)
#
#             idx = torch.LongTensor([[int(x) for x in x_idx] for x_idx in batch_fea_idxs])
#             idx = idx.to(device)
#             value = batch_fea_values.to(device, dtype=torch.float32)
#             target = batch_labels.to(device, dtype=torch.float32)
#             output = model(idx, value)
#
#             test_loss += F.binary_cross_entropy_with_logits(output, target)
#
#             pred_y.extend(list(output.cpu().numpy()))
#             true_y.extend(list(target.cpu().numpy()))
#
#         print('Roc AUC: %.5f' % roc_auc_score(y_true=np.array(true_y), y_score=np.array(pred_y)))
#         test_loss /= math.ceil(test_item_count / BATCH_SIZE)
#         print('Test set: Average loss: {:.5f}'.format(test_loss))


def train_test_model_demo(model, device, train_data_path, test_data_path, feat_dict_):
    print("Start Training Model!")

    # Sort the Train files in order
    train_filelist = ["%s%s" % (train_data_path, x) for x in os.listdir(train_data_path)]
    train_file_id = [int(re.sub('^.*[\D]', '', x)) for x in train_filelist]
    train_filelist = [train_filelist[idx] for idx in np.argsort(train_file_id)]

    # Sort the Test files in order
    test_filelist = ["%s%s" % (test_data_path, x) for x in os.listdir(test_data_path)]
    test_file_id = [int(re.sub('^.*[\D]', '', x)) for x in test_filelist]
    test_filelist = [test_filelist[idx] for idx in np.argsort(test_file_id)]

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, EPOCHS + 1):
        train_model(model, train_filelist, feat_dict_, device, optimizer, epoch)
        test_model(model, test_filelist, feat_dict_, device)


""" ************************************************************************************ """
"""                      Using Criteo DataSet to train/test Model                        """
""" ************************************************************************************ """
def train_model(model, train_filelist, feat_dict_, device, optimizer, epoch,
                use_reg_l1=False, use_reg_l2=False):
    """
    训练模型, 由于数据量过大, 如果使用PyTorch的DataLoader来load数据的话, 会很慢
    这里, 使用的方式是依次读取每一个文件, 然后进行Shuffle,
    最后, 依次拿出Batch大小的数据进行计算, 至该文件读取完毕, 则继续读取下一个文件至遍历完毕
    :param model:
    :param train_filelist:
    :param feat_dict_:
    :param device:
    :param optimizer:
    :param epoch:
    :param use_reg_l1:
    :param use_reg_l2:
    :return:
    """
    fname_idx = 0
    features_idxs, features_values, labels = None, None, None
    train_item_count = count_in_filelist_items(train_filelist)

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

        if use_reg_l1:
            for param in model.parameters():
                loss += model.reg_l1 * torch.sum(torch.abs(param))
        if use_reg_l2:
            for param in model.parameters():
                loss += model.reg_l2 * torch.sum(torch.pow(param, 2))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{} / {} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(idx), train_item_count,
                100. * batch_idx / math.ceil(int(train_item_count / BATCH_SIZE)), loss.item()))


def test_model(model, test_filelist, feat_dict_, device):
    """
    测试模型在测试集上的性能, 这里的方式基本与Train_model一样(除了不用Shuffle与梯度计算)
    :param model:
    :param test_filelist:
    :param feat_dict_:
    :param device:
    :return:
    """
    fname_idx = 0
    pred_y, true_y = [], []
    features_idxs, features_values, labels = None, None, None
    test_loss = 0
    test_item_count = count_in_filelist_items(test_filelist)
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
            output = model(idx, value, use_dropout=False)

            test_loss += F.binary_cross_entropy_with_logits(output, target)

            pred_y.extend(list(output.cpu().numpy()))
            true_y.extend(list(target.cpu().numpy()))

        print('Roc AUC: %.5f' % roc_auc_score(y_true=np.array(true_y), y_score=np.array(pred_y)))
        test_loss /= math.ceil(test_item_count / BATCH_SIZE)
        print('Test set: Average loss: {:.5f}'.format(test_loss))


def count_in_filelist_items(filelist):
    count = 0
    for fname in filelist:
        with open(fname.strip(), 'r') as fin:
            for _ in fin:
                count += 1
    return count


def get_idx_value_label(fname, feat_dict_, shuffle=True):
    """
    读取文件数据: 从一个数据文件中, 读取并得到Label, Feat_index, Feat_value值
    :param fname:
    :param feat_dict_:
    :param shuffle:
    :return:
    """
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)
    cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cont_max_ = [5775, 257675, 65535, 969, 23159456, 431037, 56311, 6047, 29019, 46, 231, 4008, 7393]
    cont_diff_ = [cont_max_[i] - cont_min_[i] for i in range(len(cont_min_))]

    def _process_line(line):
        features = line.rstrip('\n').split('\t')
        feat_idx = []
        feat_value = []

        # MinMax Normalization
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

