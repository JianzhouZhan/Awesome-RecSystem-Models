# 基于Pytorch框架实现的推荐系统的经典模型

### 1. 相关数据集
- ##### Criteo数据集
    - 整个数据集包含约4500W条记录. 每一行的第1列为Label, 表示点击与否, 
然后接下来是13个整型特征(I1-I13)以及26个离散型特征(C1-C26)
    - 数据集的下载链接为http://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/
    - 数据集下载后放置在data/Criteo/目录下
- ##### Movielens100K
    - movielens100k数据集 ，包含943个用户对于1682个影片超过10万条评分信息。推荐算法研究最常用的数据集
    - 数据集包含 ua.base, ua.test, u.item, u.user 4个文件
    - 由于数据集比较小, 这里直接提供放置在data/Movielens100K目录下了 
### 1. 实现的模型:
- ##### FM: Factorization Machine
    - 论文链接: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
    - 使用测试数据集: Movielens100K
    - 支持多分类预测
- ##### FFM: Field-aware Factorization Machine
    - 论文链接: https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
DeepFM: 
    - 使用测试数据集: Movielens100K
    - 支持多分类预测
- ##### DeepFM: Factorization-Machine based Neural Network
    - 论文链接: https://www.ijcai.org/proceedings/2017/0239.pdf
    - 使用测试数据集: Criteo
    - 根据论文的方式将数据集按9:1分成训练集+测试集
    - 先运行data/deepFM_dataProcess.py进行数据预处理, 再运行Model/Basic-DeepFM-Demo.py
    - 经过3个Epoch训练之后, AUC可以达到0.795(略低于论文的效果)
- ##### DCN: Deep&Cross Network
    - 论文链接: https://arxiv.org/pdf/1708.05123.pdf
    - 使用测试数据集: Criteo
    - 在处理数据集时, 仅简单地把数据按9:1分成训练集+测试集, 并没有按论文0.9:0.05:0.05的方式来划分
    - 先运行data/DCN_dataProcess.py进行数据预处理, 再运行Model/Basic-DCN-Demo.py
    - ###### 这里暂时还不能复现出论文的效果, 还不知是什么原因
    