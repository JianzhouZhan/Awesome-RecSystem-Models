import re
import os
import math
import pickle
import torch
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from util.train_test_util_TensorFlow import train_test_demo


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


class DeepFM(tf.keras.Model):
    def __init__(self, num_feat, num_field, dropout_deep, dropout_fm,
                 reg_l1=0.01, reg_l2=0.01, layer_sizes=[400, 400, 400], embedding_size=10):
        super().__init__()  # Python2 下使用 super(DeepFM, self).__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2                  # L1/L2正则化并没有去使用
        self.num_feat = num_feat              # denote as M
        self.num_field = num_field            # denote as F
        self.embedding_size = embedding_size  # denote as K
        self.layer_sizes = layer_sizes

        self.dropout_deep = dropout_deep
        self.dropout_fm = dropout_fm

        # first order term parameters embedding
        self.first_weights = tf.keras.layers.Embedding(num_feat, 1, embeddings_initializer='uniform')  # None * M * 1

        # 需要定义一个 Embedding
        self.feat_embeddings = tf.keras.layers.Embedding(num_feat, embedding_size, embeddings_initializer='uniform')
        # None * M * K

        # 神经网络方面的参数
        self.dense1 = tf.keras.layers.Dense(layer_sizes[0])
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.activation1 = tf.keras.layers.Activation('relu')
        self.dropout1 = tf.keras.layers.Dropout(dropout_deep[1])

        self.dense2 = tf.keras.layers.Dense(layer_sizes[1])
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.activation2 = tf.keras.layers.Activation('relu')
        self.dropout2 = tf.keras.layers.Dropout(dropout_deep[2])

        self.dense3 = tf.keras.layers.Dense(layer_sizes[2])
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.activation3 = tf.keras.layers.Activation('relu')
        self.dropout3 = tf.keras.layers.Dropout(dropout_deep[3])

        # 最后一层全连接层
        self.fc = tf.keras.layers.Dense(1, activation=None, use_bias=True)
        # self.fc = nn.Linear(num_field + embedding_size + all_dims[-1], 1)

    def call(self, feat_index, feat_value):
        feat_value = tf.expand_dims(feat_value, axis=-1)                     # None * F * 1

        # Step1: 先计算一阶线性的部分 sum_square part
        first_weights = self.first_weights(feat_index)                        # None * F * 1
        first_weight_value = tf.math.multiply(first_weights, feat_value)

        y_first_order = tf.math.reduce_sum(first_weight_value, axis=2)         # None * F
        y_first_order = tf.keras.layers.Dropout(self.dropout_fm[0])(y_first_order)  # None * F

        # Step2: 再计算二阶部分
        secd_feat_emb = self.feat_embeddings(feat_index)                      # None * F * K
        feat_emd_value = tf.math.multiply(secd_feat_emb, feat_value)          # None * F * K(广播)

        # sum_square part
        summed_feat_emb = tf.math.reduce_sum(feat_emd_value, axis=1)          # None * K
        interaction_part1 = tf.math.pow(summed_feat_emb, 2)                   # None * K

        # squared_sum part
        squared_feat_emd_value = tf.math.pow(feat_emd_value, 2)                # None * K
        interaction_part2 = tf.math.reduce_sum(squared_feat_emd_value, axis=1)  # None * K
        y_secd_order = 0.5 * tf.math.subtract(interaction_part1, interaction_part2)
        y_secd_order = tf.keras.layers.Dropout(self.dropout_fm[1])(y_secd_order)

        # Step3: Deep部分
        y_deep = tf.reshape(feat_emd_value, (-1, self.num_field * self.embedding_size))  # None * (F * K)
        y_deep = tf.keras.layers.Dropout(self.dropout_deep[0])(y_deep)

        y_deep = self.dense1(y_deep)
        y_deep = self.batch_norm1(y_deep)
        y_deep = self.activation1(y_deep)
        y_deep = self.dropout1(y_deep)

        y_deep = self.dense2(y_deep)
        y_deep = self.batch_norm2(y_deep)
        y_deep = self.activation2(y_deep)
        y_deep = self.dropout2(y_deep)

        y_deep = self.dense3(y_deep)
        y_deep = self.batch_norm3(y_deep)
        y_deep = self.activation3(y_deep)
        y_deep = self.dropout3(y_deep)

        concat_input = tf.concat((y_first_order, y_secd_order, y_deep), axis=1)
        output = self.fc(concat_input)
        return output


if __name__ == '__main__':
    train_data_path, test_data_path = AID_DATA_DIR + 'train_data/', AID_DATA_DIR + 'test_data/'
    feat_dict_ = pickle.load(open(AID_DATA_DIR + 'aid_data/feat_dict_10.pkl2', 'rb'))

    deepfm = DeepFM(num_feat=len(feat_dict_) + 1, num_field=39,
                    dropout_deep=[0.5, 0.5, 0.5, 0.5], dropout_fm=[0, 0],
                    layer_sizes=[400, 400, 400], embedding_size=10)

    train_label_path = AID_DATA_DIR + 'train_label'
    train_idx_path = AID_DATA_DIR + 'train_idx'
    train_value_path = AID_DATA_DIR + 'train_value'

    test_label_path = AID_DATA_DIR + 'test_label'
    test_idx_path = AID_DATA_DIR + 'test_idx'
    test_value_path = AID_DATA_DIR + 'test_value'

    train_test_demo(deepfm, train_label_path, train_idx_path, train_value_path, test_label_path, test_idx_path,
                    test_value_path)
