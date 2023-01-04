#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/2 下午5:41
# @Author  : xiaot

from concurrent import futures
import grpc
# from pytorch_util.conf.sentry_conf import SENTRY_DSN
# from pytorch_util.src.util.sentry_sdk_handler import SentryHandler
from pytorch_util.conf.base_model_conf import MODEL_FILE_PATH
from pytorch_util.conf.mlp_model_conf import LIST_N_FEATURES_PER_LAYER, LIST_PROB_DROPOUT, ACTIVATION_FUNCTION_HIDDEN
from pytorch_util.src.util.mlp_model_agent import MLPModelAgent
from pytorch_util.src.util.serialization import msg_unpackb, msg_packb
from pytorch_util.src.util.io_util.logging_instance import logger
from pytorch_util.conf.server_conf import RPC_SERVER_PORT
import pytorch_util.pb.TorchService_pb2_grpc as TSRPC
import pytorch_util.pb.TorchService_pb2 as TS

# build model
model = MLPModelAgent(list_n_features_per_layer=LIST_N_FEATURES_PER_LAYER,
                      list_prob_dropout=LIST_PROB_DROPOUT,
                      activation_function=ACTIVATION_FUNCTION_HIDDEN)

# load model
model.load_state_dict(file_path=MODEL_FILE_PATH)


class RequestRpc(TSRPC.TorchService):
    """pytorch_util RPC服务"""

    def ping(self, request, context):
        """
        计算服务基础服务 心跳检测
        :param request:
        :param context:
        :return: 心跳检测状态
        """
        logger.info("running ping")
        logger.info("finished ping")
        return TS.PingResponse(state='ok')

    def extract_batch(self, request, context):
        """
        批量提取模型结果。
        输入样例:
            [[0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1], [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
        输出样例:
            [1，1, 0]
        :param request:
        :param context:
        :return:
        """
        logger.info("running extract_batch")

        # get input list<list>
        bytes_list_x = request.bytes_list_x
        list_x = msg_unpackb(bytes_list_x)
        list_y = model.predict(X=list_x).tolist()
        # pack to bytes & build response
        _response = TS.ExtractBatchResponse(bytes_list_y=msg_packb(list_y))

        logger.info("finished extract_batch")

        return _response

    def extract_single(self, request, context):
        """
        单条提取模型结果。
        输入样例:
            [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
        输出样例:
            1
        :param request:
        :param context:
        :return:
        """
        logger.info("running extract_single")

        bytes_x = request.bytes_x
        x = msg_unpackb(bytes_x)
        list_x = [x]
        list_y = model.predict(X=list_x).tolist()
        y = list_y[0]
        # pack to bytes & build response
        _response = TS.ExtractSingleClassificationResponse(y=y)

        logger.info("finished extract_single")

        return _response


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=50),
                         options=[
                             ('grpc.max_send_message_length', 100 * 1024 * 1024),
                             ('grpc.max_receive_message_length', 100 * 1024 * 1024)
                         ])

    TSRPC.add_TorchServiceServicer_to_server(RequestRpc(), server)
    server.add_insecure_port('[::]:%d' % RPC_SERVER_PORT)
    server.start()
    logger.info('TorchService RPC server is activated at [::]:%d' % RPC_SERVER_PORT)
    server.wait_for_termination()


if __name__ == '__main__':
    # # sentry
    # sentry_client = SentryHandler(dsn=SENTRY_DSN)

    # serve
    serve()
