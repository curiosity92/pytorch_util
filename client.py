#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/2 下午5:41
# @Author  : xiaot

import os
import grpc
from pytorch_util.conf.server_conf import RPC_SERVER_HOST, RPC_SERVER_PORT
from pytorch_util.src.util.io_util.logging_instance import logger
from pytorch_util.src.util.serialization import msg_packb, msg_unpackb
import pytorch_util.pb.TorchService_pb2_grpc as TSRPC
import pytorch_util.pb.TorchService_pb2 as TS


def run_batch_client(list_x: list):
    """
    rpc客户端执行
    :param list_x: [[0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1]]
    :return:
    """
    with grpc.insecure_channel(RPC_SERVER_HOST + ':' + str(RPC_SERVER_PORT),
                               options=(('grpc.enable_http_proxy', 0),)) as channel:
        stub = TSRPC.TorchServiceStub(channel)

        # def ping():
        #     r = DNLPS.PingRequest(check='Are you OK?')
        #     response = stub.ping(r)
        #     if response.state == 'ok':
        #         return 'service is OK'
        #     else:
        #         return 'service has been GG'

        logger.debug('list_x: %s' % list_x)

        # build input bytes
        bytes_list_x = msg_packb(list_x)

        # extract
        _request = TS.ExtractBatchRequest(bytes_list_x=bytes_list_x)
        _response = stub.extract_batch(_request)

        # get output list<list<int/float> >
        bytes_list_y = _response.bytes_list_y
        list_y = msg_unpackb(bytes_list_y)

        return list_y


def run_single_client(x):
    """
    rpc客户端执行
    :param x: [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
    :return:
    """
    with grpc.insecure_channel(RPC_SERVER_HOST + ':' + str(RPC_SERVER_PORT),
                               options=(('grpc.enable_http_proxy', 0),)) as channel:
        stub = TSRPC.TorchServiceStub(channel)

        logger.debug('x: %s' % x)

        # build input bytes
        bytes_x = msg_packb(x)

        # extract
        _request = TS.ExtractSingleRequest(bytes_x=bytes_x)
        _response = stub.extract_single(_request)

        # get output
        y = _response.y

        return y


if __name__ == '__main__':

    # 批处理接口
    list_x = [[0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1]]
    print('list_x: %s' % list_x)
    list_y = run_batch_client(list_x)
    print('list_y: %s' % list_y)
    print('')

    # 单条处理接口
    x = [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
    print('x: %s' % x)
    y = run_single_client(x=x)
    print('y: %s' % y)
    print('')
