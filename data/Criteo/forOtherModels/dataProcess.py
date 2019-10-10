from data.Criteo.util import *

"""
[1] PaddlePaddle implementation of DeepFM for CTR prediction
    https://github.com/PaddlePaddle/models/blob/develop/PaddleRec/ctr/deepfm/data/preprocess.py
"""

if __name__ == '__main__':
    if not os.path.isdir('train_data'):
        os.mkdir('train_data')
    if not os.path.isdir('test_data'):
        os.mkdir('test_data')
    if not os.path.isdir('aid_data'):
        os.mkdir('aid_data')

    get_raw_data()
    split_data()
    get_feat_dict()

    print('Done!')

