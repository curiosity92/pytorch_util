#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/13 下午5:02
# @Author  : xiaot
import json
import os
import pandas as pd
from pkg_resources import resource_filename

from pytorch_util.src.util.io_util.logging_instance import logger

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(resource_filename(__name__, ''))), 'data', 'dataset')
logger.info('DATASET_DIR: %s' % DATASET_DIR)


def get_X_Y_from_csv(file_name, index_x, index_y):
    """
    get X, Y matrix from csv file
    :param file_name:
    :param index_x:
    :param index_y:
    :return:
    """
    file_path = os.path.join(DATASET_DIR, file_name)
    df = pd.read_csv(file_path)
    X = df.iloc[:, index_x]
    Y = df.iloc[:, index_y]

    X = [json.loads(x) for x in X]
    X = [[float(f) for f in x] for x in X]
    Y = [y for y in Y]

    return X, Y

