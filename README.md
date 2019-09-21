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
    - 支持多分类评估
- ##### FFM: Field-aware Factorization Machine
    - 论文链接: https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
DeepFM: 
    - 使用测试数据集: Movielens100K
    - 支持多分类评估