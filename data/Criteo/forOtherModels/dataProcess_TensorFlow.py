from data.Criteo.util import *

import numpy as np
"""
Data Process for FM, PNN, and DeepFM.

[1] PaddlePaddle implementation of DeepFM for CTR prediction
    https://github.com/PaddlePaddle/models/blob/develop/PaddleRec/ctr/deepfm/data/preprocess.py
"""


def get_train_test_file(file_path, feat_dict_, split_ratio=0.9):
    train_label_fout = open('train_label', 'w')
    train_value_fout = open('train_value', 'w')
    train_idx_fout = open('train_idx', 'w')
    test_label_fout = open('test_label', 'w')
    test_value_fout = open('test_value', 'w')
    test_idx_fout = open('test_idx', 'w')

    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)
    cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cont_max_ = [5775, 257675, 65535, 969, 23159456, 431037, 56311, 6047, 29019, 46, 231, 4008, 7393]
    cont_diff_ = [cont_max_[i] - cont_min_[i] for i in range(len(cont_min_))]

    def process_line_(line):
        features = line.rstrip('\n').split('\t')
        feat_idx, feat_value, label = [], [], []

        # MinMax Normalization
        for idx in continuous_range_:
            if features[idx] == '':
                feat_idx.append(0)
                feat_value.append(0.0)
            else:
                feat_idx.append(feat_dict_[idx])
                feat_value.append(round((float(features[idx]) - cont_min_[idx - 1]) / cont_diff_[idx - 1], 6))

        # 处理分类型数据
        for idx in categorical_range_:
            if features[idx] == '' or features[idx] not in feat_dict_:
                feat_idx.append(0)
                feat_value.append(0.0)
            else:
                feat_idx.append(feat_dict_[features[idx]])
                feat_value.append(1.0)
        return feat_idx, feat_value, [int(features[0])]

    with open(file_path, 'r') as fin:
        for line_idx, line in enumerate(fin):
            if line_idx % 1000000 == 0:
                print(line_idx)

            if line_idx >= EACH_FILE_DATA_NUM * 10:
                break

            feat_idx, feat_value, label = process_line_(line)

            feat_value = '\t'.join([str(v) for v in feat_value]) + '\n'
            feat_idx = '\t'.join([str(idx) for idx in feat_idx]) + '\n'
            label = '\t'.join([str(idx) for idx in label]) + '\n'

            if np.random.random() <= split_ratio:
                train_label_fout.write(label)
                train_idx_fout.write(feat_idx)
                train_value_fout.write(feat_value)
            else:
                test_label_fout.write(label)
                test_idx_fout.write(feat_idx)
                test_value_fout.write(feat_value)

        fin.close()

    train_label_fout.close()
    train_idx_fout.close()
    train_value_fout.close()
    test_label_fout.close()
    test_idx_fout.close()
    test_value_fout.close()


def get_feat_dict():
    freq_ = 10
    dir_feat_dict_ = 'feat_dict_' + str(freq_) + '.pkl2'
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)

    if os.path.exists(dir_feat_dict_):
        feat_dict = pickle.load(open(dir_feat_dict_, 'rb'))
    else:
        # print('generate a feature dict')
        # Count the number of occurrences of discrete features
        feat_cnt = Counter()
        with open('../train.txt', 'r') as fin:
            for line_idx, line in enumerate(fin):
                # for test
                if line_idx >= EACH_FILE_DATA_NUM * 10:
                    break

                if line_idx % EACH_FILE_DATA_NUM == 0:
                    print('generating feature dict', line_idx / 45000000)
                features = line.rstrip('\n').split('\t')
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
        with open('../train.txt', 'r') as fin:
            for line_idx, line in enumerate(fin):
                # get mini-sample for test
                if line_idx >= EACH_FILE_DATA_NUM * 10:
                    break

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

    return feat_dict


if __name__ == '__main__':
    feat_dict = get_feat_dict()
    get_train_test_file('../train.txt', feat_dict)
    print('Done!')

