import time
import traceback
from functools import partial
from typing import Optional, List

from tqdm import tqdm

from utils import log_utils, os_lib


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
    Usages:
        # example 1
        # print the obj
        wrapper = CallbackWrapper(
            success_callbacks=[
                StdOutCallback(FakeLogger()),
            ])

        @wrapper.process_wrap
        def func(obj, **kwargs):
            return obj

        wrapper_kwargs = wrapper.gen_kwargs(None)
        func(10, **wrapper_kwargs)

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

        wrapper_kwargs = wrapper.gen_kwargs(None)
        fun1(10, **wrapper_kwargs)
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

    def gen_kwargs(self, obj, **kwargs):
        return dict(
            callback_status={name: None for name, _ in self.success_callbacks + self.failure_callbacks}
        )

    def process_wrap(self, func):
        _func = self.on_process_start_wrap(func)
        _func = self.on_process_wrap(_func)
        _func = self.on_process_end_wrap(_func)
        return _func

    def on_process_start_end_wrap(self, func):
        _func = self.on_process_start_wrap(func)
        _func = self.on_process_end_wrap(_func)
        return _func

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
    """note, can not use for `MultiProcessDataSequential`

    Usages:
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

    def on_process_start(self, obj, callback_status={}, **kwargs):
        # set a very small delay to avoid printing the pbar when initialization
        callback_status[self.name] = tqdm(delay=1e-9)

    def on_process(self, obj: dict, callback_status={}, **kwargs):
        callback_status[self.name].update()

    def on_process_end(self, obj, callback_status={}, **kwargs):
        callback_status[self.name].close()


class TimeLoggerCallback(Module):
    def __init__(self, logger=None):
        super().__init__()
        self.logger = log_utils.get_logger(logger)

    def on_process_start(self, obj, callback_status={}, **kwargs) -> None:
        callback_status[self.name] = time.time()

    def on_process_end(self, obj, callback_status={}, **kwargs) -> None:
        st = callback_status[self.name]
        et = time.time()
        self.logger.info(f'Takes {et - st:.2f} s')


class FileCacherCallback(Module):
    """cache input and output to file"""

    def __init__(
            self, save_dir,
            cache_inputs=True, cache_outputs=True,
            save_input_keys=None, save_output_keys=None,
            **cacher_kwargs
    ):
        super().__init__()
        self.cacher = os_lib.FileCacher(save_dir, **cacher_kwargs)
        self.cache_inputs = cache_inputs
        self.cache_outputs = cache_outputs
        self.save_input_keys = save_input_keys
        self.save_output_keys = save_output_keys

    def on_process_start(self, obj, task_id=None, **kwargs) -> None:
        if not self.cache_inputs:
            return

        if self.save_input_keys:
            assert isinstance(obj, dict), 'Only dict obj support keys filter, or set `save_input_keys=None`'
            obj = {k: v for k, v in obj.items() if k in self.save_input_keys}

        self.cacher.cache_one(obj, file_stem=f'{task_id}.input')

    def on_process_end(self, obj, task_id=None, **kwargs) -> None:
        if not self.cache_outputs:
            return

        if self.save_output_keys:
            assert isinstance(obj, dict), 'Only dict obj support keys filter, or set `save_output_keys=None`'
            obj = {k: v for k, v in obj.items() if k in self.save_output_keys}

        self.cacher.cache_one(obj, file_stem=f'{task_id}.output')
