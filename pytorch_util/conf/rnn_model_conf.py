#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/7/4  11:41
# @Author  : xiaot

# rnn model layer units
import os

from pytorch_util.conf.rnn_model_name_conf import MODEL_NAME_LSTM, MODEL_NAME_GRU

EMBEDDING_DIM = 3
NUM_STEPS = 288
HIDDEN_DIM = 512  # 256
OUT_DIM = 3


# rnn model type
RNN_MODEL_TYPE = os.getenv('RNN_MODEL_TYPE', MODEL_NAME_LSTM)  # TODO gru error: "RuntimeError: Can't detach views in-place. Use detach() instead."
NUM_RNN_MODEL_LAYER = 1  # 2
IS_RNN_MODEL_BIDIRECTIONAL = False  # True
ACTIVATION_FUNCTION_HIDDEN = 'relu'

# dropout
PROB_DROPOUT = 0.1  # 0.5
