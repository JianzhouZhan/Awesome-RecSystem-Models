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
AID_DATA_DIR = '../data/Criteo/forDeepFM/'  # 辅助用途的文件路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
PyTorch implementation of DeepFM[1]

Reference:
[1] Product-based Neural Networks for User ResponsePrediction,
    Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, Jun Wang
"""

class PNN_layer(nn.Module):

    def __init__(self, num_feat, num_field, dropout_deep, deep_layer_sizes, linear_dim=10, quadratic_dim=10,
                 reg_l1=0.01, reg_l2=1e-5, embedding_size=10, product_type='inner'):
        super(PNN_layer, self).__init__()
        self.num_feat = num_feat
        self.num_field = num_field

        # Embedding
        feat_embedding = nn.Embedding(num_feat, embedding_size)
        nn.init.xavier_uniform_(feat_embedding)
        self.feat_embedding = feat_embedding

        # linear part
        linear_weights = torch.randn((linear_dim, num_field, embedding_size))
        nn.init.xavier_uniform_(linear_weights)
        self.linear_weights = linear_weights

        # quadratic part
        if product_type == 'inner':
            theta = torch.randn((quadratic_dim, num_field))
            nn.init.xavier_uniform_(theta)
        else:
            pass


