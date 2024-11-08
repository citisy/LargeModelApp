import traceback
from typing import Optional

from tqdm import tqdm

from utils import log_utils


class CallbackWrapper:
    ignore_errors = False

    def __init__(self, func, success_callbacks=None, failure_callbacks=None, **kwargs):
        self.func = func

        self.success_callbacks = []
        self.failure_callbacks = []
        self.register_success_callbacks(success_callbacks or [])
        self.register_failure_callbacks(failure_callbacks or [])

        self.name = type(self).__name__
        self.__dict__.update(kwargs)

    def register_success_callbacks(self, callbacks):
        for callback in callbacks:
            self.register_success_callback(callback)

    def register_success_callback(self, callback):
        if hasattr(callback, 'name'):
            name = callback.name
        else:
            name = type(callback).__name__
        self.success_callbacks.append((name, callback))

    def register_failure_callbacks(self, callbacks):
        for callback in callbacks:
            self.register_failure_callback(callback)

    def register_failure_callback(self, callback):
        if hasattr(callback, 'name'):
            name = callback.name
        else:
            name = type(callback).__name__
        self.failure_callbacks.append((name, callback))

    def initialize_success_callbacks(self):
        for name, callback in self.success_callbacks:
            if hasattr(callback, 'init'):
                callback.init()

    def initialize_failure_callbacks(self):
        for name, callback in self.failure_callbacks:
            if hasattr(callback, 'init'):
                callback.init()

    def __call__(self, *args, **kwargs):
        try:
            obj = self.func(*args, **kwargs)

            self.on_success(obj, **kwargs)
            return obj
        except Exception as e:
            obj = self.on_failure(e, **kwargs)

            if self.ignore_errors:
                return obj
            else:
                raise e

    def on_failure(self, e, **kwargs):
        obj = self.parse_exception(e, **kwargs)
        for name, callback in self.failure_callbacks:
            callback(obj, **kwargs)
        return obj

    def parse_exception(self, e, **kwargs):
        return e

    def on_success(self, obj, **kwargs):
        for name, callback in self.success_callbacks:
            callback(obj, **kwargs)
        return obj


class StdOutCallback:
    def __init__(self, logger=None):
        self.logger = log_utils.get_logger(logger)

    def __call__(self, obj, **kwargs):
        self.logger.info(obj)


class StdErrCallback:
    def __init__(self, logger=None):
        self.logger = log_utils.get_logger(logger)

    def __call__(self, obj, **kwargs):
        tb_info = str(traceback.format_exc()).rstrip('\n')
        self.logger.error(tb_info)


class TqdmVisCallback:
    """
    Usage:
        Sequential(
            ...,

            iter_success_callbacks=[
                TqdmVisCallback()
            ]
        )
    """

    pbar: Optional

    def init(self, obj, **kwargs):
        # avoid to print when initialization
        self.pbar = tqdm(delay=1e-9)

    def __call__(self, obj: dict, **kwargs):
        self.pbar.update()
        return obj
