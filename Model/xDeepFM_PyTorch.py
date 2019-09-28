import re
import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from time import time

EPOCHS = 5
BATCH_SIZE = 2048
AID_DATA_DIR = '../data/Criteo/forDCN/'
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
"""

class xDeepFM_layer(nn.Module):
    def __init__(self, num_dense_feat, num_sparse_field, dropout_deep, deep_layer_sizes, cin_layer_sizes,
                 reg_l1=0.01, reg_l2=0.01, embedding_size=10):
        super(xDeepFM_layer, self).__init__()
        self.num_dense_feat = num_dense_feat
        self.num_sparse_field = num_sparse_field

        self.cin_layer_sizes = cin_layer_sizes
        self.deep_layer_sizes = deep_layer_sizes

        self.input_dim = num_dense_feat + num_sparse_field * embedding_size

        #

        # Deep Part
        # Neural Network
        all_dims = [self.input_dim] + deep_layer_sizes
        for i in range(len(deep_layer_sizes)):
            setattr(self, 'linear_' + str(i + 1), nn.Linear(all_dims[i], all_dims[i + 1]))
            setattr(self, 'batchNorm_' + str(i + 1), nn.BatchNorm1d(all_dims[i + 1]))
            setattr(self, 'dropout_' + str(i + 1), nn.Dropout(dropout_deep[i + 1]))
