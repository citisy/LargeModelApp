from typing import List, Dict, Optional

from utils import log_utils, op_utils
from .callbacks import CallbackWrapper


class Module:
    # control the module is masked or applied
    mask = False
    apply = True

    callback_wrapper_ins = CallbackWrapper
    callback_wrapper_kwargs = dict()

    def __init__(self, logger=None, **kwargs):
        self.name = type(self).__name__
        self.__dict__.update(kwargs)

        self.logger = log_utils.get_logger(logger)

        self._nodes = [
            self.on_process_start,
            self.on_process,
            self.on_process_end,
        ]

        # note, not necessary to use judgment statements, but for the convenience of debugging
        if self.callback_wrapper_kwargs:
            self.add_callback()

    def add_callback(self):
        self.callback_wrapper = self.callback_wrapper_ins(**self.callback_wrapper_kwargs)
        self._process = self.callback_wrapper.process_wrap(self._process)

    @property
    def ignore_errors(self):
        if hasattr(self, 'callback_wrapper'):
            return self.callback_wrapper.ignore_errors
        else:
            return False

    @ignore_errors.setter
    def ignore_errors(self, ignore):
        if hasattr(self, 'callback_wrapper'):
            self.callback_wrapper.ignore_errors = ignore
        elif ignore:
            self.add_callback()
            self.callback_wrapper.ignore_errors = ignore

    def gen_kwargs(self, obj, **kwargs):
        return kwargs

    def __call__(self, obj, **kwargs):
        return self._process(obj, **kwargs)

    def _process(self, obj, **kwargs):
        kwargs = self.gen_kwargs(obj, **kwargs)
        for node in self._nodes:
            obj = node(obj, **kwargs)  # noqa

        return obj

    def on_process_start(self, obj, **kwargs):
        return obj

    def on_process(self, obj, **kwargs):
        return obj

    def on_process_end(self, obj, **kwargs):
        return obj

    def get_default_kwargs(self, k, kwargs: dict):
        return kwargs.get(k, self.__dict__.get(k))

    def set_default_kwargs(self, kwargs: dict):
        instance_dict = self.__dict__
        for k in log_utils.get_class_annotations(self):
            kwargs.setdefault(k, instance_dict.get(k))

    def __repr__(self):
        return self.name


class LoopModule(Module):
    """
    Usages:
        class Print(LoopModule):
            def check(self, obj, counter=None, steps=0, **kwargs):
                return counter < obj

            def on_process(self, obj, counter=None, **kwargs):
                print(counter)
                return obj

        loop = Print()
        loop(10)    # run 10 times
    """
    check_before_loop = True

    def _process(self, obj, **kwargs):
        counter: Optional[int | dict] = self.gen_counter(obj, **kwargs)  # record the check counter
        steps: int = 0  # record the loop step

        while True:
            # except multiple values error
            kwargs.update(counter=counter, steps=steps)

            if self.check_before_loop and not self.check(obj, **kwargs):
                break

            obj = super()._process(obj, **kwargs)
            counter = self.update_counter(obj, **kwargs)

            # except multiple values error
            kwargs.update(counter=counter, steps=steps)

            if not self.check_before_loop and not self.check(obj, **kwargs):
                break

            steps += 1

        return obj

    def check(self, obj, counter=None, steps=0, **kwargs) -> bool:
        """False to break the loop"""
        raise NotImplemented

    def gen_counter(self, obj, **kwargs) -> Optional[int | dict]:
        return 0

    def update_counter(self, obj, counter=0, **kwargs) -> Optional[int | dict]:
        return counter + 1


class RetryModule(Module):
    """
    Usages:
        class E(RetryModule):
            def on_process(self, obj, **kwargs):
                raise ValueError(obj)

        module = E()
        module(0)
    """

    retry_count = 3
    retry_wait = 15
    err_type = Exception

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        i = self._nodes.index(self.on_process)
        self.retry = op_utils.Retry(stdout_method=self.logger.info, count=self.retry_count, wait=self.retry_wait)
        self._nodes[i] = self.retry.add_try(err_type=self.err_type)(self._nodes[i])


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
            if isinstance(module, ModuleList):
                name = str(module)
            name = '\n'.join([' ' * 4 + _ for _ in name.split('\n')])
            s += name + '\n'
        s = s + ')'
        return s

    def module_info(self):
        s = []
        for name, module in self.modules:
            if isinstance(module, ModuleList):
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


class LoopPipeline(Pipeline, LoopModule):
    """
    Usages:
        class MyLoop(LoopPipeline):
            def check(self, obj, counter=None, steps=0, **kwargs):
                return counter < obj

        class Print(Module):
            def on_process(self, obj, counter=None, **kwargs):
                print(counter)
                return obj

        pipe = MyLoop(
            Print()
        )
        pipe(10)    # run 10 times
    """


class RetryPipeline(Pipeline, RetryModule):
    """
    Usages:
        class E(Module):
            def on_process(self, obj, **kwargs):
                raise ValueError(obj)

        pipe = RetryPipeline(
            E()
        )
        pipe(0)
    """


class SwitchPipeline(Pipeline):
    """
    Usages:
        class A(Module):
            def on_process(self, obj, **kwargs):
                print('A')
                return obj

        class B(Module):
            def on_process(self, obj, **kwargs):
                print('B')
                return obj

        class C(Module):
            def on_process(self, obj, **kwargs):
                print('C')
                return obj

        class MySwitch(SwitchPipeline):
            def switch(self, obj, **kwargs) -> str:
                return obj

        pipe = MySwitch(
            A(),
            B(),
            C()
        )

        pipe('B')
    """
    def on_process(self, obj, **kwargs):
        name = self.switch(obj, **kwargs)
        module = self.get_module(name)
        obj = module(obj, **kwargs)

        if isinstance(obj, Exception):
            raise obj

        return obj

    def switch(self, obj, **kwargs) -> str:
        """return the name of selected module"""
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

    skip_exception_return = False

    iter_callback_wrapper_ins = CallbackWrapper
    iter_callback_wrapper_kwargs = dict()

    def __init__(self, *modules, force_check=True, **kwargs):
        if force_check and not isinstance(modules[0], BaseSequentialInput):
            modules = [BaseSequentialInput()] + list(modules)

        super().__init__(*modules, **kwargs)

        # note, not necessary to use judgment statements, but for the convenience of debugging
        if self.iter_callback_wrapper_kwargs:
            self.add_iter_callback()

    def add_iter_callback(self):
        self.iter_callback_wrapper = self.iter_callback_wrapper_ins(**self.iter_callback_wrapper_kwargs)
        self._iter_result = self.iter_callback_wrapper.on_process_wrap(self._iter_result)
        self._iter = self.iter_callback_wrapper.on_process_start_end_wrap(self._iter)

    @property
    def ignore_iter_errors(self):
        if hasattr(self, 'iter_callback_wrapper'):
            return self.iter_callback_wrapper.ignore_errors
        else:
            return False

    @ignore_iter_errors.setter
    def ignore_iter_errors(self, ignore):
        if hasattr(self, 'iter_callback_wrapper'):
            self.iter_callback_wrapper.ignore_errors = ignore
        elif ignore:
            self.add_iter_callback()
            self.iter_callback_wrapper.ignore_errors = ignore

    def ignore_iter_errors_(self, ignore=True):
        self.apply_setting(ignore, 'ignore_iter_errors', 'ignore_iter_errors_')

    def on_process(self, obj, **kwargs):
        return self._iter(obj, **kwargs)

    def _iter(self, obj, **kwargs):
        _, input_module = self.modules[0]
        results = []
        for iter_obj in input_module(obj, **kwargs):
            iter_obj = self._iter_result(iter_obj, **kwargs)
            if self.skip_exception_return and isinstance(iter_obj, Exception):
                continue
            results.append(iter_obj)
        return results

    def _iter_result(self, iter_obj, **kwargs):
        return self._iter_module(iter_obj, **kwargs)

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

    def _iter(self, obj, mask_modules=(), **kwargs):
        _, input_module = self.modules[0]
        results = []
        i = 0
        iter_objs = []
        for iter_obj in input_module(obj, **kwargs):
            i += 1

            iter_objs.append(iter_obj)
            if i < self.batch_size:
                continue

            _iter_objs = self._iter_result(iter_objs, **kwargs)

            if not (self.skip_exception_return and isinstance(_iter_objs, Exception)):
                results += _iter_objs

            i = 0
            iter_objs = []

        if iter_objs:
            _iter_objs = self._iter_result(iter_objs, **kwargs)
            if not (self.skip_exception_return and isinstance(_iter_objs, Exception)):
                results += _iter_objs

        return results


class MultiProcessModuleSequential(Sequential):
    """modules parallel by multiple processes
    each module will have the same inputs
    """

    n_pool = None

    def _iter_module(self, iter_obj, **kwargs):
        from multiprocessing.pool import Pool

        pool = Pool(self.n_pool)
        processes = {}
        for name, module in self.modules[1:]:
            if self.module_control(name, module, **kwargs):
                continue

            processes[name] = pool.apply_async(module, args=(iter_obj,), kwds=kwargs)

        pool.close()
        pool.join()

        results = {}
        for name, p in processes.items():
            iter_obj = p.get()
            if self.skip_exception_return and isinstance(iter_obj, Exception):
                continue
            results[name] = iter_obj

        return self.merge_outputs(results, **kwargs)

    def merge_outputs(self, results: dict, **kwargs):
        """merge multi-process modules' returns"""
        raise NotImplemented


class MultiProcessDataSequential(Sequential):
    """data parallel by multiple processes"""
    n_pool = None

    def _iter(self, obj, **kwargs):
        from multiprocessing.pool import Pool

        pool = Pool(self.n_pool)

        _, input_module = self.modules[0]
        processes = []
        for iter_obj in input_module(obj, **kwargs):
            processes.append(pool.apply_async(self._iter_module, args=(iter_obj,), kwds=kwargs))

        pool.close()
        pool.join()

        results = []
        for p in processes:
            iter_obj = self._iter_result(p, **kwargs)
            if self.skip_exception_return and isinstance(iter_obj, Exception):
                continue
            results.append(iter_obj)

        return results

    def _iter_result(self, p, **kwargs):
        return p.get()


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

    def _iter(self, obj, **kwargs):
        _, input_module = self.modules[0]
        results = []
        threads = []
        for iter_obj in input_module(obj, **kwargs):
            threads.append(self.pool.submit(self._iter_module, iter_obj, **kwargs))

        for t in threads:
            iter_obj = self._iter_result(t, **kwargs)
            if self.skip_exception_return and isinstance(iter_obj, Exception):
                continue
            results.append(iter_obj)

        return results

    def _iter_result(self, t, **kwargs):
        return t.result()


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
