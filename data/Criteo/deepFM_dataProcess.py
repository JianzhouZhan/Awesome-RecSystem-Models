import os
import numpy
import shutil
import pickle
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
import re

EACH_FILE_DATA_NUM = 204800


def get_criteo_data(filelist):
    """
    获取文件目录下的criteo数据集
    :param filelist:
    :return:
    """
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)
    cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cont_max_ = [5775, 257675, 65535, 969, 23159456, 431037, 56311, 6047, 29019, 46, 231, 4008, 7393]
    cont_diff_ = [cont_max_[i] - cont_min_[i] for i in range(len(cont_min_))]
    print(os.getcwd())
    feat_dict_ = pickle.load(open('data/aid_data/feat_dict_10.pkl2', 'rb'))

    count = 0
    features_idxs, features_values, labels = [], [], []
    for fname in filelist:

        count += 1
        print(count)

        with open(fname.strip(), 'r') as fin:
            for line in fin:
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

                #
                for idx in categorical_range_:
                    if features[idx] == '' or features[idx] not in feat_dict_:
                        feat_idx.append(0)
                        feat_value.append(0.0)
                    else:
                        feat_idx.append(feat_dict_[features[idx]])
                        feat_value.append(1.0)

                features_idxs.append(feat_idx)
                features_values.append(feat_value)
                labels.append([int(features[0])])
    features_idxs = np.array(features_idxs)
    features_values = np.array(features_values)
    labels = np.array(labels).astype(np.int32)
    return features_idxs, features_values, labels


class CriteoDataset(Dataset):
    """
    这种方式由于速度过慢, 已经弃用
    """
    def __init__(self, filelist):
        file_idxs = [int(re.sub('[\D]', '', x)) for x in filelist]
        self.filelist = [filelist[idx] for idx in np.argsort(file_idxs)]
        self.each_file_item_num = EACH_FILE_DATA_NUM
        item_count = 0
        for fname in filelist:
            with open(fname.strip(), 'r') as fin:
                for _ in fin:
                    item_count += 1
        self.item_count = item_count

        self.continuous_range_ = range(1, 14)
        self.categorical_range_ = range(14, 40)
        self.cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cont_max_ = [5775, 257675, 65535, 969, 23159456, 431037, 56311, 6047, 29019, 46, 231, 4008, 7393]
        self.cont_diff_ = [self.cont_max_[i] - self.cont_min_[i] for i in range(len(self.cont_min_))]
        self.feat_dict_ = pickle.load(open('data/aid_data/feat_dict_10.pkl2', 'rb'))

    def __getitem__(self, idx):
        # 根据idx得到它所在的下标
        file_idx = int(idx / self.each_file_item_num)
        item_idx = idx % self.each_file_item_num

        feat_idx = []
        feat_value = []
        label = []

        fname = self.filelist[file_idx]
        with open(fname.strip(), 'r') as fin:
            count = 0
            for line_idx, line in enumerate(fin):
                count += 1
                if line_idx == item_idx:
                    features = line.rstrip('\n').split('\t')

                    # MinMax标准化连续型数据
                    for idx in self.continuous_range_:
                        if features[idx] == '':
                            feat_idx.append(0)
                            feat_value.append(0.0)
                        else:
                            feat_idx.append(self.feat_dict_[idx])
                            feat_value.append((float(features[idx]) - self.cont_min_[idx - 1]) / self.cont_diff_[idx - 1])

                    #
                    for idx in self.categorical_range_:
                        if features[idx] == '' or features[idx] not in self.feat_dict_:
                            feat_idx.append(0)
                            feat_value.append(0.0)
                        else:
                            feat_idx.append(self.feat_dict_[features[idx]])
                            feat_value.append(1.0)

                    label.append(int(features[0]))
        return np.array(feat_idx), np.array(feat_value), np.array(label).astype(np.int32)

    def __len__(self):
        return self.item_count


def get_raw_data():
    if not os.path.isdir('raw_data'):
        os.mkdir('raw_data')
    print(os.getcwd())
    fin = open('train.txt', 'r')
    fout = open('raw_data/part-0', 'w')
    for line_idx, line in enumerate(fin):
        # if line_idx >= EACH_FILE_DATA_NUM * 10:
        #     break

        if line_idx % EACH_FILE_DATA_NUM == 0 and line_idx != 0:
            fout.close()
            cur_part_idx = int(line_idx / EACH_FILE_DATA_NUM)
            fout = open('raw_data/part-' + str(cur_part_idx), 'w')
        fout.write(line)

    fout.close()
    fin.close()


def split_data():
    split_rate_ = 0.9
    dir_train_file_idx_ = 'aid_data/train_file_idx.txt'
    filelist_ = ['raw_data/part-%d' % x for x in range(len(os.listdir('raw_data')))]

    if not os.path.exists(dir_train_file_idx_):
        train_file_idx = list(
            numpy.random.choice(
                len(filelist_), int(len(filelist_) * split_rate_), False))
        with open(dir_train_file_idx_, 'w') as fout:
            fout.write(str(train_file_idx))
    else:
        with open(dir_train_file_idx_, 'r') as fin:
            train_file_idx = eval(fin.read())

    for idx in range(len(filelist_)):
        if idx in train_file_idx:
            shutil.move(filelist_[idx], 'train_data')
        else:
            shutil.move(filelist_[idx], 'test_data')


def get_feat_dict():
    freq_ = 10
    dir_feat_dict_ = 'aid_data/feat_dict_' + str(freq_) + '.pkl2'
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)

    if not os.path.exists(dir_feat_dict_):
        # print('generate a feature dict')
        # Count the number of occurrences of discrete features
        feat_cnt = Counter()
        with open('train.txt', 'r') as fin:
            for line_idx, line in enumerate(fin):
                # if line_idx >= EACH_FILE_DATA_NUM * 10:
                #     break

                if line_idx % EACH_FILE_DATA_NUM == 0:
                    print('generating feature dict', line_idx / 45000000)
                features = line.lstrip('\n').split('\t')
                for idx in categorical_range_:
                    if features[idx] == '': continue
                    feat_cnt.update([features[idx]])

        # Only retain discrete features with high frequency
        dis_feat_set = set()
        for feat, ot in feat_cnt.items():
            if ot >= freq_:
                dis_feat_set.add(feat)

        # Create a dictionary for continuous and discrete features
        feat_dict = {}
        tc = 1
        # Continuous features
        for idx in continuous_range_:
            feat_dict[idx] = tc
            tc += 1
        # Discrete features
        cnt_feat_set = set()
        with open('train.txt', 'r') as fin:
            for line_idx, line in enumerate(fin):
                # if line_idx >= EACH_FILE_DATA_NUM * 10:
                #     break

                features = line.rstrip('\n').split('\t')
                for idx in categorical_range_:
                    if features[idx] == '' or features[idx] not in dis_feat_set:
                        continue
                    if features[idx] not in cnt_feat_set:
                        cnt_feat_set.add(features[idx])
                        feat_dict[features[idx]] = tc
                        tc += 1

        # Save dictionary
        with open(dir_feat_dict_, 'wb') as fout:
            pickle.dump(feat_dict, fout)
        print('args.num_feat ', len(feat_dict) + 1)


if __name__ == '__main__':
    if not os.path.isdir('train_data'):
        os.mkdir('train_data')
    if not os.path.isdir('test_data'):
        os.mkdir('test_data')
    if not os.path.isdir('aid_data'):
        os.mkdir('aid_data')

    get_raw_data()
    split_data()
    get_feat_dict()

    print('Done!')

