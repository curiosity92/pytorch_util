# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import pytorch_util.pb.TorchService_pb2 as TorchService__pb2


class TorchServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.extract_batch = channel.unary_unary(
                '/PyTorchService.TorchService/extract_batch',
                request_serializer=TorchService__pb2.ExtractBatchRequest.SerializeToString,
                response_deserializer=TorchService__pb2.ExtractBatchResponse.FromString,
                )
        self.extract_single = channel.unary_unary(
                '/PyTorchService.TorchService/extract_single',
                request_serializer=TorchService__pb2.ExtractSingleRequest.SerializeToString,
                response_deserializer=TorchService__pb2.ExtractSingleClassificationResponse.FromString,
                )


class TorchServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def extract_batch(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def extract_single(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_TorchServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'extract_batch': grpc.unary_unary_rpc_method_handler(
                    servicer.extract_batch,
                    request_deserializer=TorchService__pb2.ExtractBatchRequest.FromString,
                    response_serializer=TorchService__pb2.ExtractBatchResponse.SerializeToString,
            ),
            'extract_single': grpc.unary_unary_rpc_method_handler(
                    servicer.extract_single,
                    request_deserializer=TorchService__pb2.ExtractSingleRequest.FromString,
                    response_serializer=TorchService__pb2.ExtractSingleClassificationResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'PyTorchService.TorchService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class TorchService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def extract_batch(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/PyTorchService.TorchService/extract_batch',
            TorchService__pb2.ExtractBatchRequest.SerializeToString,
            TorchService__pb2.ExtractBatchResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def extract_single(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/PyTorchService.TorchService/extract_single',
            TorchService__pb2.ExtractSingleRequest.SerializeToString,
            TorchService__pb2.ExtractSingleClassificationResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
