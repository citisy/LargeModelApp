import traceback
from typing import List, Dict, Optional

from utils import log_utils, op_utils


class Module:
    mask = False
    apply = True
    ignore_errors = False

    def __init__(self, logger=None, success_callbacks=None, failure_callbacks=None, **kwargs):
        self.logger = log_utils.get_logger(logger)

        self._nodes = [
            self.on_process_start,
            self.on_process,
            self.on_process_end,
        ]

        if success_callbacks is None:
            success_callbacks = []
        if failure_callbacks is None:
            failure_callbacks = []

        self.success_callbacks = []
        self.failure_callbacks = []
        self.register_success_callbacks(success_callbacks)
        self.register_failure_callbacks(failure_callbacks)

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

    def initialize_success_callbacks(self, obj, **kwargs):
        for name, callback in self.success_callbacks:
            if hasattr(callback, 'init'):
                callback.init(obj, **kwargs)

    def initialize_failure_callbacks(self, obj, **kwargs):
        for name, callback in self.failure_callbacks:
            if hasattr(callback, 'init'):
                callback.init(obj, **kwargs)

    def gen_kwargs(self, obj, **kwargs):
        return kwargs

    def __call__(self, obj, **kwargs):
        try:
            kwargs = self.gen_kwargs(obj, **kwargs)
            for node in self._nodes:
                obj = node(obj, **kwargs)  # noqa

            self.on_success(obj, **kwargs)
            return obj
        except Exception as e:
            self.on_failure(e, **kwargs)

            if not self.ignore_errors:
                raise e

    def on_process_start(self, obj, **kwargs):
        self.initialize_success_callbacks(obj, **kwargs)
        self.initialize_failure_callbacks(obj, **kwargs)
        return obj

    def on_process(self, obj, **kwargs):
        return obj

    def on_process_end(self, obj, **kwargs):
        return obj

    def parse_exception(self, e, **kwargs):
        return e

    def on_failure(self, e, **kwargs):
        tb_info = str(traceback.format_exc()).rstrip('\n')
        self.logger.error(tb_info)
        obj = self.parse_exception(e, **kwargs)
        for name, callback in self.failure_callbacks:
            callback(obj, **kwargs)
        return obj

    def on_success(self, obj, **kwargs):
        for name, callback in self.success_callbacks:
            callback(obj, **kwargs)
        return obj

    def get_default_kwargs(self, k, kwargs: dict):
        return kwargs.get(k, self.__dict__.get(k))

    def set_default_kwargs(self, kwargs: dict):
        for k in log_utils.get_class_annotations(self):
            kwargs.setdefault(k, self.__dict__.get(k))


class BaseServer(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._modules = [
            self.on_receive_start,
            *self._nodes,
            self.on_respond_end
        ]

    def on_receive_start(self, obj, **kwargs):
        return obj

    def on_respond_end(self, obj, **kwargs):
        return obj


class ModuleList(Module):
    def __init__(self, *modules, **kwargs):
        super().__init__(**kwargs)
        self.modules = []
        self.register_modules(modules)

    def register_modules(self, modules):
        for module in modules:
            self.register_module(module)

    def register_module(self, module):
        """set `module.name` as the module key
        if not a `Module` type, convert it by the rule
            list -> Sequential
            tuple -> MultiThreadSequential
            set -> MultiProcessSequential
        """
        if isinstance(module, list):
            module = Sequential(*module)
        elif isinstance(module, tuple):
            module = MultiThreadModuleSequential(*module)
        elif isinstance(module, set):
            module = MultiProcessModuleSequential(module)
        elif isinstance(module, dict):
            pass

        if hasattr(module, 'name'):
            name = module.name
        else:
            name = type(module).__name__
        self.modules.append((name, module))

    def get_module(self, name):
        for _name, module in self.modules:
            if _name == name:
                return module

    def module_control(self, name, module, mask_modules=(), apply_modules=(), **kwargs):
        """control the module is masked or applied,
        return `Ture` to mask module, `False` to apply module
        """
        # ignore `mask_modules` when `module.mask=True`
        if getattr(module, 'mask', False) or name in mask_modules:
            return True

        # ignore `apply_modules` when `module.apply=False`
        if not (getattr(module, 'apply', True) or name in apply_modules):
            return True

        return False

    def apply_setting(self, obj, func_name, cur_func_name):
        setattr(self, func_name, obj)

        for name, module in self.modules:
            if isinstance(module, (Sequential, Pipeline)):
                module.apply_setting(obj, func_name, cur_func_name)
            else:
                setattr(module, func_name, obj)

    def logger_(self, logger=None):
        self.apply_setting(log_utils.get_logger(logger), 'logger', 'logger_')

    def ignore_errors_(self, ignore=True):
        self.apply_setting(ignore, 'ignore_errors', 'ignore_errors_')

    def apply_(self, apply=True):
        self.apply_setting(apply, 'apply', 'apply_')

    def mask_(self, mask=True):
        self.apply_setting(mask, 'mask', 'mask_')

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get_module(key)
        elif isinstance(key, int):
            return self.modules[key][1]
        else:
            raise

    def __repr__(self):
        s = f'{self.name}(\n'
        for name, module in self.modules:
            if isinstance(module, (Sequential, Pipeline)):
                name = str(module)
            name = '\n'.join([' ' * 4 + _ for _ in name.split('\n')])
            s += name + '\n'
        s = s + ')'
        return s

    def module_info(self):
        s = []
        for name, module in self.modules:
            if isinstance(module, (Sequential, Pipeline)):
                name = module.module_info()
            s.append(name)
        return {self.__class__.__name__: s}


class Pipeline(ModuleList):
    """run module step by step
    the next module start process until the last module has processed all the data,
    and the output of the last module will be the input of the next module
    """

    def on_process(self, obj, **kwargs):
        for name, module in self.modules:
            if self.module_control(name, module, **kwargs):
                continue

            obj = module(obj, **kwargs)
            if isinstance(obj, Exception):
                raise obj

        return obj


class LoopPipeline(Pipeline):
    check_before_loop = True

    def on_process(self, obj, **kwargs):
        counter: Optional[int, dict] = self.gen_counter(obj, **kwargs)   # record the check counter
        steps: int = 0   # record the loop step

        while True:
            # except multiple values error
            kwargs.update(counter=counter, steps=steps)

            if self.check_before_loop and not self.check(obj, **kwargs):
                break

            obj = super().on_process(obj, **kwargs)
            counter = self.update_counter(obj, **kwargs)

            # except multiple values error
            kwargs.update(counter=counter, steps=steps)

            if not self.check_before_loop and not self.check(obj, **kwargs):
                break

            steps += 1

        return obj

    def check(self, obj, counter=None, steps=0, **kwargs):
        """False to break the loop"""
        raise NotImplemented

    def gen_counter(self, obj, **kwargs):
        return 0

    def update_counter(self, obj, counter=0, **kwargs):
        return counter + 1


class RetryPipeline(Pipeline):
    count = 3
    wait = 15
    err_type = Exception

    def __init__(self, *modules, **kwargs):
        super().__init__(*modules, **kwargs)
        i = self._nodes.index(self.on_process)
        self._nodes[i] = op_utils.Retry(stdout_method=self.logger.info, count=self.count, wait=self.wait).add_try(err_type=self.err_type)(self._nodes[i])


class SwitchPipeline(Pipeline):
    def on_process(self, obj, **kwargs):
        raise NotImplemented

    def switch(self, obj, **kwargs):
        raise NotImplemented


class MultiProcessPipeline(Pipeline):
    """modules parallel by multiple processes
    each module will have the same inputs,
    use inplace mode to return the outputs"""
    n_pool = None

    def on_process(self, obj, **kwargs):
        from multiprocessing.pool import Pool

        pool = Pool(self.n_pool)
        processes = []
        for name, module in self.modules:
            if self.module_control(name, module, **kwargs):
                continue

            processes.append(pool.apply_async(module, args=(obj,), kwds=kwargs))

        pool.close()
        pool.join()

        for p in processes:
            _obj = p.get()
            if isinstance(_obj, Exception):
                raise _obj

        # inplace mode to return the outputs
        return obj


class MultiThreadPipeline(Pipeline):
    """modules parallel by multiple threads
    each module will have the same inputs,
    use inplace mode to return the outputs"""
    n_pool = None

    def __init__(self, *modules, **kwargs):
        super().__init__(*modules, **kwargs)
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max_workers=self.n_pool)

    def on_process(self, obj, **kwargs):
        threads = []
        for name, module in self.modules:
            if self.module_control(name, module, **kwargs):
                continue

            threads.append(self.pool.submit(module, obj, **kwargs))

        for t in threads:
            _obj = t.result()
            if isinstance(_obj, Exception):
                raise _obj

        # inplace mode to return the outputs
        return obj


class Sequential(ModuleList):
    """run data step by step
    the next data start process until the last data has been processed by all the modules,
    the first module must be an instance of `BaseSequentialInput`,
    if not provided, default creates a new `BaseSequentialInput` module
    """

    skip_exception = False

    def __init__(self, *modules, force_check=True, iter_success_callbacks=None, iter_failure_callbacks=None, **kwargs):
        if force_check and not isinstance(modules[0], BaseSequentialInput):
            modules = [BaseSequentialInput()] + list(modules)

        if iter_success_callbacks is None:
            iter_success_callbacks = []
        if iter_failure_callbacks is None:
            iter_failure_callbacks = []

        self.iter_success_callbacks = []
        self.iter_failure_callbacks = []
        self.register_iter_success_callbacks(iter_success_callbacks)
        self.register_iter_failure_callbacks(iter_failure_callbacks)

        super().__init__(*modules, **kwargs)

    def register_iter_success_callbacks(self, callbacks):
        for callback in callbacks:
            self.register_iter_success_callback(callback)

    def register_iter_success_callback(self, callback):
        if hasattr(callback, 'name'):
            name = callback.name
        else:
            name = type(callback).__name__
        self.iter_success_callbacks.append((name, callback))

    def register_iter_failure_callbacks(self, callbacks):
        for callback in callbacks:
            self.register_iter_failure_callback(callback)

    def register_iter_failure_callback(self, callback):
        if hasattr(callback, 'name'):
            name = callback.name
        else:
            name = type(callback).__name__
        self.iter_failure_callbacks.append((name, callback))

    def initialize_iter_success_callbacks(self, obj, **kwargs):
        for name, callback in self.iter_success_callbacks:
            if hasattr(callback, 'init'):
                callback.init(obj, **kwargs)

    def initialize_iter_failure_callbacks(self, obj, **kwargs):
        for name, callback in self.iter_failure_callbacks:
            if hasattr(callback, 'init'):
                callback.init(obj, **kwargs)

    def on_iter_success(self, obj, **kwargs):
        for name, callback in self.iter_success_callbacks:
            callback(obj, **kwargs)
        return obj

    def on_iter_failure(self, e, **kwargs):
        tb_info = str(traceback.format_exc()).rstrip('\n')
        self.logger.error(tb_info)
        obj = self.parse_exception(e, **kwargs)
        for name, callback in self.iter_failure_callbacks:
            callback(obj, **kwargs)
        return obj

    def on_process_start(self, obj, **kwargs):
        self.initialize_iter_success_callbacks(obj, **kwargs)
        self.initialize_iter_failure_callbacks(obj, **kwargs)
        return super().on_process_start(obj, **kwargs)

    def on_process(self, obj, **kwargs):
        _, input_module = self.modules[0]
        results = []
        for iter_obj in input_module(obj, **kwargs):
            try:
                iter_obj = self._iter_module(iter_obj, **kwargs)
                self.on_iter_success(iter_obj, **kwargs)
            except Exception as e:
                self.on_iter_failure(e, **kwargs)
                if not self.skip_exception:
                    raise e
            results.append(iter_obj)
        return results

    def _iter_module(self, iter_obj, **kwargs):
        for name, module in self.modules[1:]:
            if self.module_control(name, module, **kwargs):
                continue

            iter_obj = module(iter_obj, **kwargs)
            if isinstance(iter_obj, Exception):
                raise iter_obj
        return iter_obj


class BatchSequential(Sequential):
    """run batch data step by step
    the next batch data start process until the last batch data has been processed by all the modules,
    and must have an input layer which returns an iterable obj
    """

    batch_size = 1

    def on_process(self, obj, mask_modules=(), **kwargs):
        _, input_module = self.modules[0]
        results = []
        i = 0
        iter_objs = []
        for iter_obj in input_module(obj, **kwargs):
            i += 1

            iter_objs.append(iter_obj)
            if i < self.batch_size:
                continue

            results = self._iter(iter_objs, results, **kwargs)
            i = 0
            iter_objs = []

        if iter_objs:
            results = self._iter(iter_objs, results, **kwargs)

        return results

    def _iter(self, iter_objs, results, **kwargs):
        try:
            iter_objs = self._iter_module(iter_objs, **kwargs)
            self.on_iter_success(iter_objs, **kwargs)
            results += iter_objs

        except Exception as e:
            self.on_iter_failure(e, **kwargs)
            if not self.skip_exception:
                raise e

        return results


class MultiProcessModuleSequential(Sequential):
    """modules parallel by multiple processes
    each module will have the same inputs,
    use inplace mode to return the outputs"""

    n_pool = None

    def _iter_module(self, iter_obj, **kwargs):
        from multiprocessing.pool import Pool

        pool = Pool(self.n_pool)
        for name, module in self.modules[1:]:
            if self.module_control(name, module, **kwargs):
                continue

            pool.apply_async(module, args=(iter_obj,), kwds=kwargs)

        pool.close()
        pool.join()

        # inplace mode to return the outputs
        return iter_obj


class MultiProcessDataSequential(Sequential):
    """data parallel by multiple processes"""
    n_pool = None

    def on_process(self, obj, **kwargs):
        from multiprocessing.pool import Pool

        pool = Pool(self.n_pool)

        _, input_module = self.modules[0]
        results = []
        processes = []
        for iter_obj in input_module(obj, **kwargs):
            processes.append(pool.apply_async(self._iter_module, args=(iter_obj,), kwds=kwargs))

        pool.close()
        pool.join()

        for p in processes:
            try:
                iter_obj = p.get()
                results.append(iter_obj)
                self.on_iter_success(iter_obj, **kwargs)
            except Exception as e:
                self.on_iter_failure(e, **kwargs)
                if not self.skip_exception:
                    raise e

        return results


class MultiThreadModuleSequential(Sequential):
    """module parallel by multiple threads,
    each module will have the same inputs,
    use inplace mode to return the outputs"""
    n_pool = None

    def __init__(self, *modules, **kwargs):
        super().__init__(*modules, **kwargs)
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max_workers=self.n_pool)

    def _iter_module(self, iter_obj, **kwargs):
        threads = []
        for name, module in self.modules[1:]:
            if self.module_control(name, module, **kwargs):
                continue

            threads.append(self.pool.submit(module, iter_obj, **kwargs))

        for t in threads:
            _iter_obj = t.result()
            if isinstance(_iter_obj, Exception):
                raise _iter_obj

        # inplace mode to return the outputs
        return iter_obj


class MultiThreadDataSequential(Sequential):
    """data parallel by multiple threads"""
    n_pool = None

    def __init__(self, *modules, **kwargs):
        super().__init__(*modules, **kwargs)
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max_workers=self.n_pool)

    def on_process(self, obj, **kwargs):
        _, input_module = self.modules[0]
        results = []
        threads = []
        for iter_obj in input_module(obj, **kwargs):
            threads.append(self.pool.submit(self._iter_module, iter_obj, **kwargs))

        for t in threads:
            try:
                iter_obj = t.result()
                results.append(iter_obj)
                self.on_iter_success(iter_obj, **kwargs)
            except Exception as e:
                self.on_iter_failure(e, **kwargs)
                if not self.skip_exception:
                    raise e

        return results


class BaseSequentialInput(Module):
    """default input module for Sequential
    do nothing, just return an iterable obj"""

    def on_process(self, objs: list, **kwargs):
        for obj in objs:
            yield obj


class DictSequentialInput(BaseSequentialInput):
    """input module for Sequential
    apply dict obj as input objs"""

    def __init__(self, var_keys: list = [], const_keys: list = []):
        super().__init__()
        self.var_keys = var_keys
        self.const_keys = const_keys

    def on_process(self, objs: Dict[str, List], **kwargs):
        var_keys = self.var_keys or objs.keys()
        var_objs = {k: objs[k] for k in var_keys}
        const_objs = {k: objs[k] for k in self.const_keys}

        for vs in zip(var_objs.values()):
            obj = {k: v for k, v in zip(var_objs.keys(), vs)}
            obj.update(const_objs)
            yield obj


class BasePipelineInput(Module):
    """default input module for Sequential
    do nothing, just return the inputs obj"""

    def on_process(self, objs, *args, **kwargs):
        return objs


class ListPipelineInput(BasePipelineInput):
    """input module for Pipeline
    apply list obj as input obj"""

    def __init__(self, select_keys: list = None):
        super().__init__()
        self.select_keys = select_keys

    def on_process(self, objs: List[Dict], *args, **kwargs) -> Dict[str, List]:
        results = {}
        for obj in objs:
            for k, v in obj.items():
                if self.select_keys is not None and k in self.select_keys:
                    results.setdefault(k, []).append(v)
                else:
                    results[k] = v
        return results
