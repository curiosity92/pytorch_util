# -*- coding: utf-8 -*


import logging

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

logger = logging.getLogger(__name__)
sentry_logging = LoggingIntegration(
    level=logging.INFO,  # Capture info and above as breadcrumbs
    event_level=logging.ERROR  # Send errors as events
)


class SentryHandler:

    def __init__(self, dsn,
                 before_send=None,
                 traces_sampler=None,
                 attach_stacktrace=True,
                 ignore_exceptions=None):
        self.dsn = dsn
        self._sentry_ignore_errors = []
        self._sentry_ignore_errors_tuple = tuple()
        if ignore_exceptions:
            # 忽略错误类型
            self.sentry_ignore_errors(ignore_exceptions)
        self.sentry_init(before_send=before_send,
                         traces_sampler=traces_sampler,
                         attach_stacktrace=attach_stacktrace)

    def sentry_init(self, before_send=None,
                    traces_sampler=None,
                    attach_stacktrace=True):

        """
        项目初始化
         :param before_send:
        :param traces_sampler:
        :param attach_stacktrace: 打印堆栈信息
        :return:
        """
        if not self.dsn:
            return
        if not before_send:
            before_send = self._sentry_before_send

        if not traces_sampler:
            traces_sampler = self._sentry_traces_sampler
        # All of this is already happening by default!

        sentry_sdk.init(
            dsn=self.dsn,
            integrations=[sentry_logging],
            before_send=before_send,
            traces_sampler=traces_sampler,
            attach_stacktrace=attach_stacktrace
        )

    def _sentry_before_send(self, event, hint):
        # modify event here
        ignore_status = self._sentry_ignore_errors_handler(hint)
        if ignore_status:
            return

        event = self.handler_before_send(event, hint)

        return event

    def _sentry_ignore_errors_handler(self, hint: dict):
        if not self._sentry_ignore_errors:
            return False
        if 'log_record' not in hint:
            return False
        try:
            log_record = hint['log_record']
            msg = log_record.msg
            if isinstance(msg, self._sentry_ignore_errors_tuple):
                return True
        except Exception as e:
            logger.info(e)
        return False

    def handler_before_send(self, event, hint):
        # 外部重构
        return event

    def sentry_ignore_errors(self, exceptions):

        if isinstance(exceptions, list):
            self._sentry_ignore_errors.extend(exceptions)
        else:
            self._sentry_ignore_errors.append(exceptions)

        self._sentry_ignore_errors_tuple = tuple(self._sentry_ignore_errors)

    def _sentry_traces_sampler(self, sampling_context):
        # Examine provided context data (including parent decision, if any)
        # along with anything in the global namespace to compute the sample rate
        # or sampling decision for this transaction
        # 返回 0-1  返回0.5  该事件有50%左右会被提交  默认100% 提交
        return 1

    def handler_traces_sampler(self, sampling_context):
        # Examine provided context data (including parent decision, if any)
        # along with anything in the global namespace to compute the sample rate
        # or sampling decision for this transaction
        # 返回 0-1  返回0.5  该事件有50%左右会被提交  默认100% 提交
        # 外部重构
        return 1
