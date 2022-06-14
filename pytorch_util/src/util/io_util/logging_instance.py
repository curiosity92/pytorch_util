# -*- coding:utf-8 -*-
# author: xiaot
# time: 2020/5/22 上午10:20

import logging
import os
from pkg_resources import resource_filename

from pytorch_util.src.util.io_util.logging_util import LoggingUtil

MODULE_NAME = 'pytorch_util'
LOG_FILE_DIR = os.getenv('PYTORCH_UTIL_LOG_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(resource_filename(__name__, '')))), 'log'))
# print(f'LOG_FILE_DIR: {LOG_FILE_DIR}')
LOG_FILE_PATH = os.path.join(LOG_FILE_DIR, f'{MODULE_NAME}.log')
# print(f'LOG_FILE_PATH: {LOG_FILE_PATH}')
str_log_level = os.getenv('PYTORCH_UTIL_LOG_LEVEL', 'DEBUG').upper()
if str_log_level == 'INFO':
    LOG_LEVEL = logging.INFO
elif str_log_level == 'DEBUG':
    LOG_LEVEL = logging.DEBUG
elif str_log_level == 'WARN':
    LOG_LEVEL = logging.WARN
elif str_log_level == 'ERROR':
    LOG_LEVEL = logging.ERROR
else:
    assert False

if not os.path.exists(LOG_FILE_DIR):
    os.makedirs(LOG_FILE_DIR)
#     print(f'log dir automatically created: {LOG_FILE_PATH}')

logger = LoggingUtil.get_logger(module_name=MODULE_NAME, log_file_path=LOG_FILE_PATH, log_level=LOG_LEVEL)
