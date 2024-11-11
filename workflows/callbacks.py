import traceback
from functools import partial
from typing import Optional, List

from tqdm import tqdm

from utils import log_utils


class Module:
    def __init__(self, **kwargs):
        self.name = type(self).__name__
        self.__dict__.update(kwargs)

    def __call__(self, obj, **kwargs):
        self.on_process(obj, **kwargs)

    def on_process_start(self, obj, **kwargs) -> None:
        pass

    def on_process(self, obj, **kwargs) -> None:
        pass

    def on_process_end(self, obj, **kwargs) -> None:
        pass


class CallbackWrapper:
    """
    Usage:
        # example 1
        # print the obj
        wrapper = CallbackWrapper(
            success_callbacks=[
                StdOutCallback(FakeLogger()),
            ])

        @wrapper.process_wrap
        def func(obj, **kwargs):
            return obj

        func(10)

        # example 2
        # visualize the pbar
        wrapper = CallbackWrapper(
            success_callbacks=[
                TqdmVisCallback()
            ])

        @wrapper.on_process_start_end_wrap
        def fun1(obj, **kwargs):
            for i in range(obj):
                fun2(i, **kwargs)

        @wrapper.on_process_wrap
        def fun2(obj, **kwargs):
            time.sleep(random.random())
            return obj

        fun1(10)
    """
    ignore_errors = False

    def __init__(self, success_callbacks=None, failure_callbacks=None, **kwargs):
        self.success_callbacks = []
        self.failure_callbacks = []
        self.register_success_callbacks(success_callbacks or [])
        self.register_failure_callbacks(failure_callbacks or [])

        self.name = type(self).__name__
        self.__dict__.update(kwargs)

    def register_success_callbacks(self, callbacks: List[Module]):
        for callback in callbacks:
            self.register_success_callback(callback)

    def register_success_callback(self, callback: Module):
        if hasattr(callback, 'name'):
            name = callback.name
        else:
            name = type(callback).__name__
        self.success_callbacks.append((name, callback))

    def register_failure_callbacks(self, callbacks: List[Module]):
        for callback in callbacks:
            self.register_failure_callback(callback)

    def register_failure_callback(self, callback: Module):
        if hasattr(callback, 'name'):
            name = callback.name
        else:
            name = type(callback).__name__
        self.failure_callbacks.append((name, callback))

    def process_wrap(self, func):
        def wrapper(obj, **kwargs):
            _func = self.on_process_start_wrap(func)
            _func = self.on_process_wrap(_func)
            _func = self.on_process_end_wrap(_func)

            return _func(obj, **kwargs)

        return wrapper

    def on_process_start_end_wrap(self, func):
        def wrapper(obj, **kwargs):
            _func = self.on_process_start_wrap(func)
            _func = self.on_process_end_wrap(_func)
            return _func(obj, **kwargs)

        return wrapper

    def on_process_start_wrap(self, func):
        return partial(self.on_process_start, func)

    def on_process_start(self, func, obj, **kwargs):
        for name, callback in self.success_callbacks:
            if hasattr(callback, 'on_process_start'):
                callback.on_process_start(obj, **kwargs)

        for name, callback in self.failure_callbacks:
            if hasattr(callback, 'on_process_start'):
                callback.on_process_start(obj, **kwargs)

        return func(obj, **kwargs)

    def on_process_wrap(self, func):
        return partial(self.on_process, func)

    def on_process(self, func, obj, **kwargs):
        try:
            obj = func(obj, **kwargs)

            self.on_success(obj, **kwargs)
            return obj
        except Exception as e:
            obj = self.on_failure(e, **kwargs)

            if self.ignore_errors:
                return obj
            else:
                raise e

    def on_process_end_wrap(self, func):
        return partial(self.on_process_end, func)

    def on_process_end(self, func, obj, **kwargs):
        obj = func(obj, **kwargs)

        for name, callback in self.success_callbacks:
            if hasattr(callback, 'on_process_end'):
                callback.on_process_end(obj, **kwargs)

        for name, callback in self.failure_callbacks:
            if hasattr(callback, 'on_process_end'):
                callback.on_process_end(obj, **kwargs)

        return obj

    def on_failure(self, e: Exception, **kwargs):
        obj = self.parse_exception(e, **kwargs)
        for name, callback in self.failure_callbacks:
            callback(obj, **kwargs)
        return obj

    def parse_exception(self, e: Exception, **kwargs):
        """return something specially to transfer to all failure callbacks and to replace the normal returns"""
        return e

    def on_success(self, obj, **kwargs):
        for name, callback in self.success_callbacks:
            callback(obj, **kwargs)
        return obj


class StdOutCallback(Module):
    def __init__(self, logger=None):
        super().__init__()
        self.logger = log_utils.get_logger(logger)

    def on_process(self, obj, **kwargs):
        self.logger.info(obj)


class StdErrCallback(Module):
    def __init__(self, logger=None):
        super().__init__()
        self.logger = log_utils.get_logger(logger)

    def on_process(self, obj, **kwargs):
        tb_info = str(traceback.format_exc()).rstrip('\n')
        self.logger.error(tb_info)


class TqdmVisCallback(Module):
    """
    Usage:
        # use in workflow
        Sequential(
            ...,
            iter_callback_wrapper_kwargs=dict(
                success_callbacks=[
                    TqdmVisCallback()
                ]
            )
        )
    """

    pbar: Optional

    def on_process_start(self, obj, **kwargs):
        # set a very small delay to avoid printing the pbar when initialization
        self.pbar = tqdm(delay=1e-9)

    def on_process(self, obj: dict, **kwargs):
        self.pbar.update()

    def on_process_end(self, obj, **kwargs):
        self.pbar.close()
