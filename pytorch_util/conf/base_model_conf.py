#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/7/4  10:13
# @Author  : xiaot
import os
import torch
from pkg_resources import resource_filename

from pytorch_util.src.util.io_util.logging_instance import logger


# gpu acceleration
IS_USING_CUDA = torch.cuda.is_available()
logger.info("IS_USING_CUDA: %s" % IS_USING_CUDA)

# model dir
MODEL_DIR = os.path.join(os.path.dirname(resource_filename(__name__, '')), 'data', 'model')

# model file path
MODEL_FILE_NAME = os.getenv('MODEL_FILE_NAME', 'model.state_dict')
MODEL_FILE_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
logger.info('MODEL_FILE_PATH: %s' % MODEL_FILE_PATH)
