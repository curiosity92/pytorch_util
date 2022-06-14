#!/usr/bin/env python
# -*- coding: utf-8 -*

import json
import pickle
import logging

try:
    import umsgpack
except ImportError as e:
    umsgpack = None

try:
    import msgpack
except ImportError as e:
    msgpack = None


try:
    import orjson
except ImportError as e:
    orjson = None

try:
    import simdjson
except ImportError as e:
    simdjson = None

try:
    import ujson
except ImportError as e:
    ujson = None

from datetime import datetime
from struct import pack, unpack


logger = logging.getLogger(__name__)


def str_pack(data):
    """
    将字符串打包成二进制
    :param data: str
    :return:
    """
    raw = data.encode('utf-8')
    strlen = len(raw)
    res = pack('>{}s'.format(strlen), raw)
    # res = data.decode
    return res


def str_unpack(data):
    """
    将二进制数据解包
    :param data: bytes
    :return:
    """
    strlen = len(data)

    res = unpack('>{}s'.format(strlen), data)
    return res[0].decode('utf-8')


def int_pack(data, msg_len):
    """
    打包整型数据为二进制
    :param data: bytes
    :return:
    """

    _len = msg_len
    if _len == 1:
        fmt = '>B'
    elif _len == 2:
        fmt = '>H'
    elif _len == 4:
        fmt = '>I'
    elif _len == 8:
        fmt = '>Q'
    else:
        fmt = ''

    res = pack(fmt, data)
    return res


def int_unpack(data):
    """
    将二进制数据解包
    :param data: bytes
    :return:
    """

    _len = len(data)
    if _len == 1:
        fmt = '>B'
    elif _len == 2:
        fmt = '>H'
    elif _len == 4:
        fmt = '>I'
    elif _len == 8:
        fmt = '>Q'
    else:
        fmt = ''

    res = unpack(fmt, data)
    return res[0]


def umsg_packb(data):
    """

    :param data:
    :return:
    """
    return umsgpack.packb(data)


def umsg_unpackb(data):
    """

    :param data:
    :return:
    """
    return umsgpack.unpackb(data)


def msgpack_decode_datetime(obj):
    if '__datetime__' in obj:
        obj = datetime.strptime(obj["as_str"], "%Y%m%dT%H:%M:%S.%f")
    return obj


def msgpack_encode_datetime(obj):
    if isinstance(obj, datetime):
        return {'__datetime__': True, 'as_str': obj.strftime("%Y%m%dT%H:%M:%S.%f")}
    return obj


def msg_packb(data, default=None, use_bin_type=True):
    """

    :param data:
    :return:
    """
    # 解决 detatime
    try:
        default = default or msgpack_encode_datetime
        r = msgpack.packb(data, default=default, use_bin_type=use_bin_type)
    except TypeError as e:
        err = str(e)
        # 自动处理时间类型问题
        if not default and "can not serialize 'datetime.datetime'" in err:
            r = msgpack.packb(data, default=msgpack_encode_datetime, use_bin_type=use_bin_type)
        else:
            raise e

    return r


def msg_unpackb(data, object_hook=None, raw=False):
    """

    :param data:
    :return:
    """
    # 解决 detatime
    try:
        object_hook = object_hook or msgpack_decode_datetime
        r = msgpack.unpackb(data, object_hook=object_hook, raw=raw)
    except TypeError as e:
        # err = str(e)
        # if not object_hook and  "a bytes-like object is required":
        #     r = msgpack.unpackb(data, object_hook=msgpack_decode_datetime, raw=raw)
        # else:
        #     raise e
        raise e
    return r


def pickle_packb(data):
    """

    :param data:
    :return: bytes
    """
    return pickle.dumps(data)


def pickle_unpackb(data):
    """

    :param data: bytes
    :return:
    """
    return pickle.loads(data)


def json_packb(data, *args, **kwargs):
    """

    :param data:
    :return: bytes
    """
    return json.dumps(data, *args, **kwargs).encode('utf-8')


def json_unpackb(data, *args, **kwargs):
    """

    :param data: bytes
    :return:
    """
    return json.loads(data.decode('utf-8'), *args, **kwargs)


def ujson_packb(data, *args, **kwargs):
    """

    :param data:
    :return: bytes
    """
    return ujson.dumps(data, *args, **kwargs).encode('utf-8')


def ujson_unpackb(data, *args, **kwargs):
    """

    :param data: bytes
    :return:
    """
    return ujson.loads(data.decode('utf-8'), *args, **kwargs)


def orjson_packb(data, *args, **kwargs):
    """

    :param data:
    :return: bytes
    """
    return orjson.dumps(data, *args, **kwargs)


def orjson_unpackb(data, *args, **kwargs):
    """

    :param data: bytes
    :return:
    """
    return orjson.loads(data.decode('utf-8'), *args, **kwargs)


def simdjson_packb(data, *args, **kwargs):
    """

    :param data:
    :return: bytes
    """

    # 直接调用系统json
    return simdjson.dumps(data, *args, **kwargs).encode('utf-8')


def simdjson_unpackb(data, *args, **kwargs):
    """

    :param data: bytes
    :return:
    """
    return simdjson.loads(data.decode('utf-8'), *args, **kwargs)







