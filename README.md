- 运行环境：python 3
- 第三方工具：jieba, gensim, numpy, Levenshtein, xgboost
- 在命令行下运行lab.py,需要四个参数：
- 前三个参数分别为训练集，测试集，以及结果文件的路径。要求训练集也要包含0,1表示，否则读入数据会出错。
- 第四个参数为在使用新的训练集时设为1，代表重新计算训练集。使用旧训练集时设为0，从文件中读入之前的计算结果
如:
```
python lab.py data/training.data data/develop.data data/score 0
```