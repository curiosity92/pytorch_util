#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/9 下午2:28
# @Author  : xiaot
import os

from pytorch_util.conf.optimizer_name_conf import *

OPTIMIZER_NAME = os.getenv('OPTIMIZER_NAME', OPTIMIZER_NAME_ADAM)
LEARNING_RATE = os.getenv('LEARNING_RATE', 0.01)
BATCH_SIZE_TRAIN = os.getenv('BATCH_SIZE_TRAIN', 2000)
BATCH_SIZE_VALIDATION = os.getenv('BATCH_SIZE_VALIDATION', 5000)
BATCH_SIZE_TEST = os.getenv('BATCH_SIZE_TEST', 5000)
N_EPOCHS = os.getenv('N_EPOCHS', 35)

