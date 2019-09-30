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

    def __init__(self):
        super(PNN_layer, self).__init__()
        pass
