import os
import numpy as np
from torch.utils.data.dataset import random_split
import sys  # 导入sys模块
sys.setrecursionlimit(20000)  # 将默认的递归深度修改为3000

def totaldataset(root):
    train_txt = "data/food/train.txt"
    eval_txt = "data/food/val.txt"

    train_data = []
    eval_data = []
    with open(root + train_txt, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            train_data.append([os.path.join(root, line.strip().split()[0]), int(line.strip().split()[1])])

    with open(root + eval_txt, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            eval_data.append([os.path.join(root, line.strip().split()[0]), int(line.strip().split()[1])])

    num_train = int(len(eval_data) * 25 / 100)
    # num_train = 0
    print('num_train from valid:' + str(num_train))
    split_train, split_valid = random_split(eval_data, [num_train, len(eval_data) - num_train])
    return split_train + train_data, split_valid


if __name__ == '__main__':
    split_train, split_valid = totaldataset('/ceph/home/yangsc21/kaggle/')

    print(split_train)
