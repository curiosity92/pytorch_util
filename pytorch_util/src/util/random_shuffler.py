#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/13 下午5:25
# @Author  : xiaot
import random
random.seed(1)


def shuffle_2_lists(list1, list2):
    """
    shuffle 2 lists in zipped fashion
    :param list1:
    :param list2:
    :return:
    """

    assert len(list1) == len(list2)
    list_index = list(range(len(list1)))
    random.shuffle(list_index)
    list1 = [list1[index] for index in list_index]
    list2 = [list2[index] for index in list_index]
    return list1, list2
