import pickle
import tensorflow as tf
from util.train_model_util_TensorFlow import train_test_model_demo

AID_DATA_DIR = '../data/Criteo/forOtherModels/'  # 辅助用途的文件路径

"""
TensorFlow 2.0 implementation of Product-based Neural Network[1]

Reference:
[1] Product-based Neural Networks for User ResponsePrediction,
    Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, Jun Wang
[2] Tensorflow implementation of PNN
    https://github.com/Atomu2014/product-nets
"""

class PNN_layer(tf.keras.Model):

    def __init__(self, num_feat, num_field, dropout_deep, deep_layer_sizes, product_layer_dim=10,
                 reg_l1=0.01, reg_l2=1e-5, embedding_size=10, product_type='outer'):
        super().__init__()   # Python2 下使用 super(PNN_layer, self).__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.num_feat = num_feat                                   # Denoted as
        self.num_field = num_field                                 # Denoted as N
        self.product_layer_dim = product_layer_dim                 # Denoted as D1
        self.dropout_deep = dropout_deep

        # Embedding
        feat_embeddings = tf.keras.layers.Embedding(num_feat, embedding_size, embeddings_initializer='uniform')
        self.feat_embeddings = feat_embeddings

        initializer = tf.initializers.GlorotUniform()

        # linear part
        self.linear_weights = tf.Variable(initializer(shape=(product_layer_dim, num_field, embedding_size)))

        # quadratic part
        self.product_type = product_type
        if product_type == 'inner':
            self.theta = tf.Variable(initializer(shape=(product_layer_dim, num_field)))  # D1 * N
        else:
            self.quadratic_weights = tf.Variable(initializer(shape=(product_layer_dim, embedding_size,
                                                                    embedding_size)))   # D1 * M * M

        # fc layer
        self.deep_layer_sizes = deep_layer_sizes

        # 神经网络方面的参数
        for i in range(len(deep_layer_sizes)):
            setattr(self, 'dense_' + str(i), tf.keras.layers.Dense(deep_layer_sizes[i]))
            setattr(self, 'batchNorm_' + str(i), tf.keras.layers.BatchNormalization())
            setattr(self, 'activation_' + str(i), tf.keras.layers.Activation('relu'))
            setattr(self, 'dropout_' + str(i), tf.keras.layers.Dropout(dropout_deep[i]))

        # last layer
        self.fc = tf.keras.layers.Dense(1, activation=None, use_bias=True)

    def call(self, feat_index, feat_value):
        # embedding part
        feat_embedding = self.feat_embeddings(feat_index)          # Batch * N * M

        # linear part
        lz = tf.einsum('bnm,dnm->bd', feat_embedding, self.linear_weights)  # Batch * D1

        # quadratic part
        if self.product_type == 'inner':
            theta = tf.einsum('bnm,dn->bdnm', feat_embedding, self.theta)   # Batch * D1 * N * M
            lp = tf.einsum('bdnm,bdnm->bd', theta, theta)
        else:
            embed_sum = tf.reduce_sum(feat_embedding, axis=1)
            p = tf.einsum('bm,bn->bmn', embed_sum, embed_sum)
            lp = tf.einsum('bmn,dmn->bd', p, self.quadratic_weights)  # Batch * D1

        y_deep = tf.concat((lz, lp), axis=1)
        y_deep = tf.keras.layers.Dropout(self.dropout_deep[0])(y_deep)

        for i in range(len(self.deep_layer_sizes)):
            y_deep = getattr(self, 'dense_' + str(i))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(i))(y_deep)
            y_deep = getattr(self, 'activation_' + str(i))(y_deep)
            y_deep = getattr(self, 'dropout_' + str(i))(y_deep)

        output = self.fc(y_deep)
        return output


if __name__ == '__main__':
    feat_dict_ = pickle.load(open(AID_DATA_DIR + 'aid_data/feat_dict_10.pkl2', 'rb'))

    pnn = PNN_layer(num_feat=len(feat_dict_) + 1, num_field=39, dropout_deep=[0.5, 0.5, 0.5],
                    deep_layer_sizes=[400, 400], product_layer_dim=10,
                    reg_l1=0.01, reg_l2=1e-5, embedding_size=10, product_type='outer')

    train_label_path = AID_DATA_DIR + 'train_label'
    train_idx_path = AID_DATA_DIR + 'train_idx'
    train_value_path = AID_DATA_DIR + 'train_value'

    test_label_path = AID_DATA_DIR + 'test_label'
    test_idx_path = AID_DATA_DIR + 'test_idx'
    test_value_path = AID_DATA_DIR + 'test_value'

    train_test_model_demo(pnn, train_label_path, train_idx_path, train_value_path, test_label_path, test_idx_path,
                          test_value_path)
