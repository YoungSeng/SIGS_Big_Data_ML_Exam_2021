import os
import numpy as np
from torch.utils.data.dataset import random_split
import sys  # 导入sys模块
sys.setrecursionlimit(20000)  # 将默认的递归深度修改为3000

def totaldataset(root):
    train_txt = "data/food/train.txt"
    eval_txt = "data/food/val.txt"

    totaldata = []
    with open(root + train_txt, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            totaldata.append([os.path.join(root, line.strip().split()[0]), int(line.strip().split()[1])])

    with open(root + eval_txt, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            totaldata.append([os.path.join(root, line.strip().split()[0]), int(line.strip().split()[1])])

    num_train = int(len(totaldata) * 0.8)
    print('num_train:' + str(num_train))
    split_train, split_valid = random_split(totaldata, [num_train, len(totaldata) - num_train])

    return split_train, split_valid


if __name__ == '__main__':
    split_train, split_valid = totaldataset('/ceph/home/yangsc21/kaggle/')

    print(split_train)
