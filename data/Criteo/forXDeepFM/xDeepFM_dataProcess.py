import os
import numpy
import shutil
import pickle
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
import re
import pandas as pd
import sys, csv, math
from collections import defaultdict

EACH_FILE_DATA_NUM = 204800

"""
[1] PaddlePaddle implemantation of DeepFM for CTR prediction
    https://github.com/PaddlePaddle/models/blob/develop/PaddleRec/ctr/deepfm/data/preprocess.py
"""


def get_raw_data():
    if not os.path.isdir('raw_data'):
        os.mkdir('raw_data')
    print(os.getcwd())
    fin = open('../train.txt', 'r')
    fout = open('raw_data/part-0', 'w')
    for line_idx, line in enumerate(fin):
        # get mini-sample for test
        # if line_idx >= EACH_FILE_DATA_NUM * 20:
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


def scan_train_txt(filename):
    feat_cnt = defaultdict(lambda: 0)
    count = 0
    for row in csv.DictReader(open(filename)):
        count += 1
        # if count >= EACH_FILE_DATA_NUM * 20:
        #     break
        for key, val in row.items():
            if 'C' in key:
                if val == '':
                    feat_cnt[str(key) + '#' + 'absence'] += 1
                else:
                    feat_cnt[str(key) + '#' + str(val)] += 1
    return feat_cnt


def get_feat(filename, feat_cnt):
    T = 4
    featSet = set()
    count = 0
    for row in csv.DictReader(open(filename)):
        count += 1
        if count % EACH_FILE_DATA_NUM == 0:
            print('generating feature dict', count / 45000000)

        for key, val in row.items():
            if 'I' in key and key != "Id":
                if val == '':
                    featSet.add(str(key) + '#' + 'absence')
                else:
                    val = int(float(val))
                    if val > 2:
                        val = int(math.log(float(val)) ** 2)
                    else:
                        val = 'SP' + str(val)
                    featSet.add(str(key) + '#' + str(val))
                continue
            if 'C' in key:
                if val == '':
                    feat = str(key) + '#' + 'absence'
                else:
                    feat = str(key) + '#' + str(val)
                if feat_cnt[feat] > T:
                    featSet.add(feat)
                else:
                    featSet.add(str(key) + '#' + str(feat_cnt[feat]))
                continue

    featIndex = dict()
    for index, feat in enumerate(featSet, start=1):
        featIndex[feat] = index
    print('feat dict num:', len(featIndex))

    return featIndex


if __name__ == '__main__':
    train_csv_file = '../input.csv'
    if not os.path.isdir('train_data'):
        os.mkdir('train_data')
    if not os.path.isdir('test_data'):
        os.mkdir('test_data')
    if not os.path.isdir('aid_data'):
        os.mkdir('aid_data')

    get_raw_data()
    split_data()

    reader = pd.read_csv('../train.txt', delimiter='\t', header=None, chunksize=1024 * 1024)
    for idx, r in enumerate(reader):
        if idx == 0:
            r.columns = ['Label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13',
                         'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
                         'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']
            r.to_csv(train_csv_file, mode='a', header=True, index=False)
        else:
            r.to_csv(train_csv_file, mode='a', header=False, index=False)

    feat_cnt = scan_train_txt(train_csv_file)
    featIndex = get_feat(train_csv_file, feat_cnt)

    # Save dictionary
    freq_ = 4
    dir_feat_dict_ = 'aid_data/feat_dict_' + str(freq_) + '.pkl2'
    with open(dir_feat_dict_, 'wb') as fout:
        pickle.dump(featIndex, fout)

    feat_cnt_dict_ = 'aid_data/feat_cnt_' + str(freq_) + '.pkl2'
    feat_cnt_dict = {}
    for k, v in feat_cnt.items():
        feat_cnt_dict[k] = v
    with open(feat_cnt_dict_, 'wb') as fout:
        pickle.dump(feat_cnt_dict, fout)
    print('args.num_feat ', len(featIndex) + 1)
    print('Done!')

