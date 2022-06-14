#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/14 上午10:13
# @Author  : xiaot
import torch
from torch.utils.data import Dataset

from pytorch_util.conf.mlp_model_conf import LIST_N_FEATURES_PER_LAYER


class TorchData(Dataset):
    def __init__(self, X, Y):
        """

        :param X: example [[0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1], ...]
        :param Y:[1,0,...]
        """
        self.X = torch.tensor(X).view(-1, LIST_N_FEATURES_PER_LAYER[0])  # 350000 * 13
        self.Y = torch.tensor(Y).view(-1)  # 350000,
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len
