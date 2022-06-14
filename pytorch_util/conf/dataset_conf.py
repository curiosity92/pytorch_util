#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/13 下午5:29
# @Author  : xiaot


# training set split ratio
RATIO_TRAINING_SET = 0.7
RATIO_VALIDATION_SET = 0.2
RATIO_TEST_SET = 0.1
assert abs(RATIO_TRAINING_SET + RATIO_VALIDATION_SET + RATIO_TEST_SET - 1.0) < 1e-6
