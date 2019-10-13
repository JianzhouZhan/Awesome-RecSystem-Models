import pickle
import torch
import torch.nn as nn
from util.train_test_model_util import train_test_model_demo

AID_DATA_DIR = '../data/Criteo/forOtherModels/'  # 辅助用途的文件路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


""" ************************************************************************************ """
"""                                          FFM层                                       """
""" ************************************************************************************ """
class FFM_layer(nn.Module):
    def __init__(self, num_feat, num_field, reg_l1=1e-4, reg_l2=1e-4, embedding_size=10):
        super(FFM_layer, self).__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.embedding_size = embedding_size
        pass


if __name__ == '__main__':
    train_data_path, test_data_path = AID_DATA_DIR + 'train_data/', AID_DATA_DIR + 'test_data/'
    feat_dict_ = pickle.load(open(AID_DATA_DIR + 'aid_data/feat_dict_10.pkl2', 'rb'))
    fm = FFM_layer(num_feat=len(feat_dict_) + 1, num_field=39, reg_l2=1e-5, embedding_size=10).to(DEVICE)
    train_test_model_demo(fm, DEVICE, train_data_path, test_data_path, feat_dict_)