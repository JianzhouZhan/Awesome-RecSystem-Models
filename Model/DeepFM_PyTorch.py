import re
import os
import math
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from util.train_test_model_util import train_test_model_demo

EPOCHS = 10
BATCH_SIZE = 2048
AID_DATA_DIR = '../data/Criteo/forOtherModels/'  # 辅助用途的文件路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
PyTorch implementation of DeepFM[1]

Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He
[2] Tensorflow implementation of DeepFM for CTR prediction 
    https://github.com/ChenglongChen/tensorflow-DeepFM 
[3] PaddlePaddle implemantation of DeepFM for CTR prediction
    https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr/deepfm
"""

class DeepFM(nn.Module):
    def __init__(self, num_feat, num_field, dropout_deep, dropout_fm,
                 reg_l1=0.01, reg_l2=0.01, layer_sizes=[400, 400, 400], embedding_size=10):
        super(DeepFM, self).__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2                  # L1/L2正则化并没有去使用
        self.num_feat = num_feat              # denote as M
        self.num_field = num_field            # denote as F
        self.embedding_size = embedding_size  # denote as K
        self.layer_sizes = layer_sizes

        self.dropout_deep = dropout_deep
        self.dropout_fm = dropout_fm

        # first order term parameters embedding
        self.first_weights = nn.Embedding(num_feat, 1)  # None * M * 1
        nn.init.xavier_uniform_(self.first_weights.weight)

        # 需要定义一个 Embedding
        self.feat_embeddings = nn.Embedding(num_feat, embedding_size)  # None * M * K
        nn.init.xavier_uniform_(self.feat_embeddings.weight)

        # 神经网络方面的参数
        all_dims = [self.num_field * self.embedding_size] + layer_sizes
        for i in range(1, len(layer_sizes) + 1):
            setattr(self, 'linear_' + str(i), nn.Linear(all_dims[i - 1], all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout_deep[i]))

        # 最后一层全连接层
        self.fc = nn.Linear(num_field + embedding_size + all_dims[-1], 1)

    def forward(self, feat_index, feat_value):
        feat_value = torch.unsqueeze(feat_value, dim=2)                       # None * F * 1

        # Step1: 先计算一阶线性的部分 sum_square part
        first_weights = self.first_weights(feat_index)                        # None * F * 1
        first_weight_value = torch.mul(first_weights, feat_value)
        y_first_order = torch.sum(first_weight_value, dim=2)                  # None * F
        y_first_order = nn.Dropout(self.dropout_fm[0])(y_first_order)         # None * F

        # Step2: 再计算二阶部分
        secd_feat_emb = self.feat_embeddings(feat_index)                      # None * F * K
        feat_emd_value = secd_feat_emb * feat_value                           # None * F * K(广播)

        # sum_square part
        summed_feat_emb = torch.sum(feat_emd_value, 1)                        # None * K
        interaction_part1 = torch.pow(summed_feat_emb, 2)                     # None * K

        # squared_sum part
        squared_feat_emd_value = torch.pow(feat_emd_value, 2)                 # None * K
        interaction_part2 = torch.sum(squared_feat_emd_value, dim=1)          # None * K

        y_secd_order = 0.5 * torch.sub(interaction_part1, interaction_part2)
        y_secd_order = nn.Dropout(self.dropout_fm[1])(y_secd_order)

        # Step3: Deep部分
        y_deep = feat_emd_value.reshape(-1, self.num_field * self.embedding_size)  # None * (F * K)
        y_deep = nn.Dropout(self.dropout_deep[0])(y_deep)

        for i in range(1, len(self.layer_sizes) + 1):
            y_deep = getattr(self, 'linear_' + str(i))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(i))(y_deep)
            y_deep = F.relu(y_deep)
            y_deep = getattr(self, 'dropout_' + str(i))(y_deep)

        concat_input = torch.cat((y_first_order, y_secd_order, y_deep), dim=1)
        output = self.fc(concat_input)
        return output


if __name__ == '__main__':
    train_data_path, test_data_path = AID_DATA_DIR + 'train_data/', AID_DATA_DIR + 'test_data/'
    feat_dict_ = pickle.load(open(AID_DATA_DIR + 'aid_data/feat_dict_10.pkl2', 'rb'))

    deepfm = DeepFM(num_feat=len(feat_dict_) + 1, num_field=39,
                    dropout_deep=[0.5, 0.5, 0.5, 0.5], dropout_fm=[0, 0],
                    layer_sizes=[400, 400, 400], embedding_size=10).to(DEVICE)

    train_test_model_demo(deepfm, DEVICE, train_data_path, test_data_path, feat_dict_)
