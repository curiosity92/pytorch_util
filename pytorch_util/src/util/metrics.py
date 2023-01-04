#!/usr/bin/python3
# -*- coding: utf-8 -*-


def accuracy_numerator(y_hat, y, mask=None):
    """compute correct number"""

    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    cmp_num = cmp.type(y.dtype)

    if mask is not None:
        # mask invalid part
        cmp_num[~mask] = 0

    ret = float(cmp_num.sum())
    return ret
