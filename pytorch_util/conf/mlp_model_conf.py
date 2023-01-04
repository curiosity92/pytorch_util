#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/9 下午2:24
# @Author  : xiaot
import json
import os
import torch

from pytorch_util.conf.activation_name_conf import ACTIVATION_NAME_RELU, SET_ACTIVATION_NAME
from pytorch_util.src.util.io_util.logging_instance import logger

# model layers
# each layer's #features (including input layer and output layer)
# [in_features, hidden_features_1, hidden_features_2, ..., out_features]
# note: 784 = 28 * 28
LIST_N_FEATURES_PER_LAYER = json.loads(
    os.getenv('LIST_N_FEATURES_PER_LAYER',
              '[13, 32, 16, 2]'  # '[784, 50, 50, 10]'
              ))
logger.info('LIST_N_FEATURES_PER_LAYER: %s' % LIST_N_FEATURES_PER_LAYER)

# dropouts
# each hidden layer's dropout(0 <= dropout probability <= 1)
# [prob_dropout_hidden_layer_1, prob_dropout_hidden_layer_2, ...]
LIST_PROB_DROPOUT = json.loads(os.getenv('LIST_PROB_DROPOUT', '[0, 0]'))
assert (not LIST_PROB_DROPOUT) or \
       (sum([1 if 0 <= p <= 1 else 0 for p in LIST_PROB_DROPOUT]) == len(LIST_PROB_DROPOUT) and \
        len(LIST_N_FEATURES_PER_LAYER) == len(LIST_PROB_DROPOUT) + 2)
logger.info('LIST_PROB_DROPOUT: %s' % LIST_PROB_DROPOUT)

# activation
ACTIVATION_FUNCTION_HIDDEN = os.getenv('ACTIVATION_FUNCTION_HIDDEN',
                                       ACTIVATION_NAME_RELU)
assert ACTIVATION_FUNCTION_HIDDEN in SET_ACTIVATION_NAME
logger.info('ACTIVATION_FUNCTION_HIDDEN: %s' % ACTIVATION_FUNCTION_HIDDEN)
