import numpy as np
from scipy.sparse import csr
import pandas as pd
import torch

def get_csr_mat_from_df(df, user_match_dict=None, item_match_dict=None):
    """
    从DataFrame中得到csr格式的表示, 同时返回对应关系
    :param df:
    :return:
    """
    if 'user' not in df.columns or 'item' not in df.columns:
        return None, user_match_dict, item_match_dict

    # Step1: 以及对应的数据, 来构造数据集
    data = np.ones(len(df.index))

    # 需要得到一种对应关系, 如 user_id为 "123456", 所对应的下标为0, item_id为"111111", 所对应的下标为0
    if user_match_dict is None:
        user_match_dict = dict()
    if item_match_dict is None:
        item_match_dict = dict()

    # 得到行下标/列下标
    rows, cols = [], []
    for idx in df['user']:
        match_id = user_match_dict.get(str(idx), len(user_match_dict))
        user_match_dict[str(idx)] = match_id
        rows.append(match_id)

    for idx in df['item']:
        match_id = item_match_dict.get(str(idx), len(item_match_dict))
        item_match_dict[str(idx)] = match_id
        cols.append(match_id)

    row_num, col_num = len(user_match_dict), len(item_match_dict)
    return csr.csr_matrix((data, (np.array(rows), np.array(cols))), shape=(row_num, col_num)), \
           user_match_dict, item_match_dict

#
# def train_FM_model_demo():
#
#     # Step1: 导入数据
#     cols = ['user', 'item', 'rating', 'timestamp']
#     train = pd.read_csv('../data/FM-Data/ua.base', delimiter='\t', names=cols)
#     test = pd.read_csv('../data/FM-Data/ua.test', delimiter='\t', names=cols)
#     y_train, y_test = np.array(train['rating']), np.array(test['rating'])
#
#     # Step2: 把数据装换成常规的格式
#     x_train, user_match_dict, item_match_dict = get_csr_mat_from_df(train)
#     x_test, user_match_dict, item_match_dict = get_csr_mat_from_df(test, user_match_dict, item_match_dict)
#     x_train, x_test = x_train.todense(), x_test.todense()
#
#     # FM模型
#     fm = FM_layer(10, 5)
#
#     # 定义损失函数还有优化器
#     criterion = nn.CrossEntropyLoss()
#     optm = torch.optim.Adam(fm.parameters())
#
#     # 开始训练
#     for i in range(EPOCHS):
#         fm.train()
#
#         x = torch.from_numpy(x_train).float()
#         y = torch.from_numpy(y_train).long()
#         y_hat = fm(x)
#         loss = criterion(y_hat, y)
#         optm.zero_grad()
#         loss.backward()
#         optm.step()
#
#         if (i + 1) % 100 == 0:
#             fm.eval()
#             test_in = torch.from_numpy(x_test).float()
#             test_l = torch.from_numpy(y_test).long()
#             test_out = fm(test_in)
#             accu = get_accu(test_out, test_l)
#             print("Epoch:{}, Loss:{:.4f}, Accuracy: {:.2f}".format(i + 1, loss.item(), accu))
