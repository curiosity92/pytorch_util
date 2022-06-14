#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/20 下午3:52
# @Author  : xiaot
import os

RPC_SERVER_HOST = os.getenv('RPC_SERVER_HOST', 'localhost')
RPC_SERVER_PORT = int(os.getenv('RPC_SERVER_PORT', '42000'))
