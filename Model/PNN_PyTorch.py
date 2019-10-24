import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.train_model_util_PyTorch import train_test_model_demo

AID_DATA_DIR = '../data/Criteo/forOtherModels/'  # 辅助用途的文件路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
PyTorch implementation of Product-based Neural Network[1]

Reference:
[1] Product-based Neural Networks for User ResponsePrediction,
    Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, Jun Wang
[2] Tensorflow implementation of PNN
    https://github.com/Atomu2014/product-nets
"""

class PNN_layer(nn.Module):

    def __init__(self, num_feat, num_field, dropout_deep, deep_layer_sizes, product_layer_dim=10,
                 reg_l1=0.01, reg_l2=1e-5, embedding_size=10, product_type='outer'):
        super().__init__()  # Python2 下使用 super(PNN_layer, self).__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.num_feat = num_feat                                   # Denoted as
        self.num_field = num_field                                 # Denoted as N
        self.product_layer_dim = product_layer_dim                 # Denoted as D1
        self.dropout_deep = dropout_deep

        # Embedding
        feat_embeddings = nn.Embedding(num_feat, embedding_size)
        nn.init.xavier_uniform_(feat_embeddings.weight)
        self.feat_embeddings = feat_embeddings

        # linear part
        linear_weights = torch.randn((product_layer_dim, num_field, embedding_size))   # D1 * N * M
        nn.init.xavier_uniform_(linear_weights)
        self.linear_weights = nn.Parameter(linear_weights)

        # quadratic part
        self.product_type = product_type
        if product_type == 'inner':
            theta = torch.randn((product_layer_dim, num_field))        # D1 * N
            nn.init.xavier_uniform_(theta)
            self.theta = nn.Parameter(theta)
        else:
            quadratic_weights = torch.randn((product_layer_dim, embedding_size, embedding_size))  # D1 * M * M
            nn.init.xavier_uniform_(quadratic_weights)
            self.quadratic_weights = nn.Parameter(quadratic_weights)

        # fc layer
        self.deep_layer_sizes = deep_layer_sizes
        all_dims = [self.product_layer_dim + self.product_layer_dim] + deep_layer_sizes
        for i in range(1, len(deep_layer_sizes) + 1):
            setattr(self, 'linear_' + str(i), nn.Linear(all_dims[i - 1], all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout_deep[i]))

        # last layer
        self.fc = nn.Linear(deep_layer_sizes[-1], 1)

    def forward(self, feat_index, feat_value):
        # embedding part
        feat_embedding = self.feat_embeddings(feat_index)          # Batch * N * M

        # linear part
        lz = torch.einsum('bnm,dnm->bd', feat_embedding, self.linear_weights)  # Batch * D1

        # quadratic part
        if self.product_type == 'inner':
            theta = torch.einsum('bnm,dn->bdnm', feat_embedding, self.theta)            # Batch * D1 * N * M
            lp = torch.einsum('bdnm,bdnm->bd', theta, theta)
        else:
            embed_sum = torch.sum(feat_embedding, dim=1)
            p = torch.einsum('bm,bm->bmm', embed_sum, embed_sum)
            lp = torch.einsum('bmm,dmm->bd', p, self.quadratic_weights)        # Batch * D1

        y_deep = torch.cat((lz, lp), dim=1)
        y_deep = nn.Dropout(self.dropout_deep[0])(y_deep)

        for i in range(1, len(self.deep_layer_sizes) + 1):
            y_deep = getattr(self, 'linear_' + str(i))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(i))(y_deep)
            y_deep = F.relu(y_deep)
            y_deep = getattr(self, 'dropout_' + str(i))(y_deep)

        output = self.fc(y_deep)
        return output


if __name__ == '__main__':
    train_data_path, test_data_path = AID_DATA_DIR + 'train_data/', AID_DATA_DIR + 'test_data/'
    feat_dict_ = pickle.load(open(AID_DATA_DIR + 'aid_data/feat_dict_10.pkl2', 'rb'))
    pnn = PNN_layer(num_feat=len(feat_dict_) + 1, num_field=39, dropout_deep=[0.5, 0.5, 0.5],
                    deep_layer_sizes=[400, 400], product_layer_dim=10,
                    reg_l1=0.01, reg_l2=1e-5, embedding_size=10, product_type='inner').to(DEVICE)
    train_test_model_demo(pnn, DEVICE, train_data_path, test_data_path, feat_dict_)
