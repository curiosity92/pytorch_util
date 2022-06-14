# -*- coding: utf-8 -*-
# author：xiaot
# time ：2020/5/22  9:42

import logging.handlers


class LoggingUtil:

    @staticmethod
    def get_logger(module_name, log_file_path, log_level):

        handler = logging.handlers.RotatingFileHandler(
            log_file_path, maxBytes=20 * 1024 * 1024, backupCount=10)  # 实例化handler
        fmt = '%(name)s\t%(levelname)s\t%(filename)s:%(lineno)s\t%(asctime)s\t%(message)s'
        formatter = logging.Formatter(fmt)  # 实例化formatter
        handler.setFormatter(formatter)  # 为handler添加formatter

        logger = logging.getLogger(module_name)  # 获取logger
        logger.addHandler(handler)  # 为logger添加handler
        logger.setLevel(log_level)
        logger.propagate = False

        return logger
