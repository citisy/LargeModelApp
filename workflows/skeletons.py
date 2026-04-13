from contextlib import contextmanager
from typing import List, Dict, Optional

from tqdm import tqdm, asyncio

from utils import log_utils, op_utils, configs, os_lib
from . import callbacks

base_module_tables = op_utils.RegisterTables()


@base_module_tables.add_register()
class Module:
    # control the module is masked or applied
    mask = False  # this value will not be changed usually, unless want to mask the module forever
    apply = True
    allow_start = False  # allow the module to be started in a workflow or not
    allow_end = False  # allow the module to be ended in a workflow or not
    inplace = False

    callback_wrapper_ins = callbacks.CallbackWrapper
    callback_wrapper_kwargs = dict()

    name: str

    def __init__(self, logger=None, success_callbacks=None, failure_callbacks=None, **kwargs):
        from . import bundled

        if not hasattr(self, 'name'):
            self.name = type(self).__name__

        self.__dict__.update(kwargs)

        self.logger = log_utils.get_logger(logger)
        self.visualize = bundled.Visualizer()

        self._nodes = [
            self.on_process_start,
            self.on_process,
            self.on_process_end,
        ]

        if success_callbacks:
            self.callback_wrapper_kwargs = configs.ConfigObjParse.merge_dict(self.callback_wrapper_kwargs, dict(success_callbacks=success_callbacks))

        if failure_callbacks:
            self.callback_wrapper_kwargs = configs.ConfigObjParse.merge_dict(self.callback_wrapper_kwargs, dict(failure_callbacks=failure_callbacks))

        # note, not necessary to use judgment statements, but for the convenience of debugging
        self.callback_wrapper = callbacks.FakeCallbackWrapper()
        if self.callback_wrapper_kwargs:
            self.add_callback()

    @classmethod
    def from_configs(cls, cfgs, *args, name=None, **kwargs):
        """
        Usage:
            from workflows import skeletons

            class A(skeletons.Pipeline):
                pass

            class AA(skeletons.Module):
                pass

            class AB(skeletons.Module):
                pass

            cfgs = dict(
                A=dict(),
                AA=dict(),
                AB=dict()
            )
            m = A.from_configs(cfgs, AA(), AB())
        """
        if name:
            pass
        elif hasattr(cls, 'name'):
            name = cls.name
        else:
            name = cls.__name__

        config = cfgs.get(name, {})
        config = configs.ConfigObjParse.merge_dict(config, kwargs)

        cls.config = config

        return cls(*args, cfgs=cfgs, name=name, **config)

    @classmethod
    def from_structure_dict(cls, structure_dict, register_tables):
        """

        Args:
            structure_dict (dict): {"name": "", "config": dict(), "modules": []}
            register_tables:

        Usage:
            from workflows import skeletons
            from utils import op_utils

            register_tables = op_utils.RegisterTables()

            @register_tables.add_register()
            class A(skeletons.Pipeline):
                pass

            @register_tables.add_register()
            class AA(skeletons.Module):
                pass

            @register_tables.add_register()
            class AB(skeletons.Module):
                pass

            structure_dict = dict(
                name='A',
                config=dict(),
                modules=[
                    dict(name='AA', config=dict()),
                    dict(name='AB', config=dict()),
                ]
            )
            m = skeletons.Module.from_structure_dict(structure_dict, register_tables)
        """
        name = structure_dict['name']
        module_cls = register_tables.get(name) or base_module_tables.get(name)
        assert module_cls is not None, f'Module `{name}` is not found in register_tables'
        modules = [cls.from_structure_dict(m, register_tables) for m in structure_dict.get('modules', [])]
        module = module_cls(*modules, **structure_dict.get('config', {}))

        return module

    def to_structure_dict(self):
        d = dict(name=self.name, config=getattr(self, 'config', {}))
        return d

    @classmethod
    def from_structure_array(cls, structure_array, register_tables, cfgs={}):
        """

        Args:
            structure_array (str| tuple): (name, [structure_array])
                e.g. "m" | ("m1", ["mm1", 'mm2', ...])
            register_tables:
            cfgs (dict)

        Usage:
            from workflows import skeletons
            from utils import op_utils

            register_tables = op_utils.RegisterTables()

            @register_tables.add_register()
            class A(skeletons.Pipeline):
                pass

            @register_tables.add_register()
            class AA(skeletons.Module):
                pass

            @register_tables.add_register()
            class AB(skeletons.Module):
                pass

            structure_array = ('A', ['AA', ('A', ['AA']), 'AB'])
            cfgs = dict(
                A=dict(),
                AA=dict(),
                AB=dict()
            )
            m = skeletons.Module.from_module_names(structure_array, register_tables, cfgs)
        """
        if isinstance(structure_array, str):
            name = structure_array
            module_cls = register_tables.get(name) or base_module_tables.get(name)
            module = module_cls.from_configs(cfgs)

        elif isinstance(structure_array, tuple):
            name, modules_names = structure_array
            module_cls = register_tables.get(name) or base_module_tables.get(name)
            modules = []
            for module_name in modules_names:
                modules.append(cls.from_structure_array(module_name, register_tables, cfgs))
            module = module_cls.from_configs(cfgs, *modules)

        else:
            raise NotImplementedError

        return module

    def to_structure_array(self):
        structure_array = self.name
        cfgs = {
            self.name: getattr(self, 'config', {})
        }
        return structure_array, cfgs

    @classmethod
    def from_structure_file(cls, file_path, register_tables):
        """
        Args:
            file_path (str): after parse the file, would like to get a dict for `from_dict`
            register_tables (dict | op_utils.RegisterTables):
        """
        dic = os_lib.loader.auto_load(file_path)
        assert isinstance(dic, dict), f'after parse the file, would like to get a `dict` not the `{type(dic)}`'
        return cls.from_structure_dict(dic, register_tables)

    def add_callback(self):
        if isinstance(self.callback_wrapper, callbacks.FakeCallbackWrapper):
            self.callback_wrapper = self.callback_wrapper_ins(module_name=self.name, **self.callback_wrapper_kwargs)
            self._process = self.callback_wrapper.process_wrap(self._process)

    @property
    def ignore_errors(self):
        return self.callback_wrapper.ignore_errors

    @ignore_errors.setter
    def ignore_errors(self, ignore):
        self.add_callback()
        self.callback_wrapper.ignore_errors = ignore

    def gen_kwargs(self, obj, **kwargs):
        return kwargs

    def __call__(self, obj, **kwargs):
        # todo, `gen_kwargs` does not be wrapped in callback
        kwargs = configs.ConfigObjParse.merge_dict(kwargs, self.callback_wrapper.gen_kwargs(obj, **kwargs), inplace=True)
        kwargs = self.gen_kwargs(obj, **kwargs)
        return self._process(obj, **kwargs)

    def _process(self, obj, **kwargs):
        for node in self._nodes:
            _obj = node(obj, **kwargs)  # noqa
            if not self.inplace:
                obj = _obj

        return obj

    def on_process_start(self, obj, **kwargs):
        return obj

    def on_process(self, obj, **kwargs):
        return obj

    def on_process_end(self, obj, **kwargs):
        return obj

    def get_default_kwargs(self, k, kwargs: dict, default=None):
        return kwargs.get(k, getattr(self, k, default))

    def set_default_kwargs(self, kwargs: dict, default=None):
        for k in log_utils.get_class_annotations(self):
            kwargs.setdefault(k, getattr(self, k, default))

    def module_info(self):
        return self.visualize.module_info(self)

    def __repr__(self):
        return self.visualize.str(self)

    def flow_chat(self, *args, **kwargs):
        """
        Usage:
            module = Module()
            module.flow_chat(filename, format='png')
        """
        return self.visualize.flow_chat(self, *args, **kwargs)


@base_module_tables.add_register()
class AsyncModule(Module):
    async def on_process(self, obj, **kwargs):
        return super().on_process(obj, **kwargs)

    async def on_process_start(self, obj, **kwargs):
        return super().on_process_start(obj, **kwargs)

    async def on_process_end(self, obj, **kwargs):
        return super().on_process_end(obj, **kwargs)

    async def __call__(self, obj, **kwargs):
        # todo, `gen_kwargs` does not be wrapped in callback
        kwargs.update(self.callback_wrapper.gen_kwargs(obj, **kwargs), inplace=True)
        kwargs = self.gen_kwargs(obj, **kwargs)
        return await self._process(obj, **kwargs)

    async def _process(self, obj, **kwargs):
        for node in self._nodes:
            _obj = await node(obj, **kwargs)  # noqa
            if not self.inplace:
                obj = _obj

        return obj


@base_module_tables.add_register()
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


@base_module_tables.add_register()
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
    raise_type = type(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.retry = op_utils.Retry(count=self.retry_count, wait=self.retry_wait)
        self._process = self.retry.add_try(err_context=self.err_context, err_type=self.err_type, raise_type=self.raise_type)(self._process)

    @contextmanager
    def err_context(self, e=None, i=None, **kwargs):
        msg = '{name}[{task_id}] error occur: "{e}", sleep {wait} seconds, and then retry!'
        msg = msg.format(e=e, wait=self.retry_wait, name=self.name, **kwargs)
        self.logger.error(msg)
        try:
            yield
        finally:
            msg = '{name}[{task_id}] {i}th process!'
            msg = msg.format(i=i + 2, name=self.name, **kwargs)
            self.logger.info(msg)


@base_module_tables.add_register()
class IgnoreExceptionModule(Module):
    err_type = Exception
    raise_type = type(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ignore_exception = op_utils.IgnoreException(stdout_method=self.logger.info)
        self._process = self.ignore_exception.add_ignore(err_type=self.err_type, raise_type=self.raise_type, err_fn=self.err_fn)(self._process)

    def err_fn(self, obj, **kwargs):
        return obj


@base_module_tables.add_register()
class SkipModule(Module):
    def _process(self, obj, **kwargs):
        if self.skip(obj, **kwargs):
            return obj

        else:
            return super()._process(obj, **kwargs)

    def skip(self, obj, **kwargs) -> bool:
        """True to skip the process, and False to access the process"""
        return False


@base_module_tables.add_register()
class ThreadsLimitModule(Module):
    n_pool = 1  # often use to control a single thread

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max_workers=self.n_pool)

    def _process(self, obj, **kwargs):
        t = self.pool.submit(super()._process, obj, **kwargs)
        obj = t.result()
        if isinstance(obj, Exception):
            raise obj
        return obj


class ModuleList(Module):
    def __init__(self, *modules, **kwargs):
        super().__init__(**kwargs)
        self.modules = []
        self.register_modules(modules)

    @staticmethod
    def _check(name, module, include=(), exclude=()):
        flag = True
        if include:
            for cls in include:
                if isinstance(cls, str):
                    if name == cls:
                        break
                elif isinstance(module, cls):
                    break
            else:
                flag = False

        if exclude:
            for cls in exclude:
                if isinstance(cls, str):
                    if name == cls:
                        flag = False
                elif isinstance(module, cls):
                    flag = False

        return flag

    def register_modules(self, modules):
        for module in modules:
            self.register_module(module)

    def register_module(self, module, name=None):
        """set `module.name` as the module key"""
        if name:
            pass
        elif hasattr(module, 'name'):
            name = module.name
        else:
            name = type(module).__name__
        self.modules.append((name, module))

    def replace_module(self, module, name=None, count=None, recursive=False):
        if name:
            pass
        elif hasattr(module, 'name'):
            name = module.name
        else:
            name = type(module).__name__

        count = count or float('inf')
        c = 0
        for i, (n, m) in enumerate(self.modules):
            if n == name:
                self.modules[i] = (name, module)
                c += 1
                if c >= count:
                    break

            if recursive and isinstance(m, ModuleList):
                m.replace_module(module, name, count - c, recursive)

    def get_module(self, name: str | int, default=None, recursive=False) -> Module:
        if isinstance(name, int):
            return self.modules[name][1]
        else:
            for _name, module in self.modules:
                if _name == name:
                    return module
            else:
                if recursive:
                    for _name, module in self.modules:
                        if isinstance(module, ModuleList):
                            m = module.get_module(name, default, recursive)
                            if m:
                                return m

        return default

    def get_modules(self, include=(), exclude=(), recursive=False):
        modules = []
        if self._check(self.name, self, include, exclude):
            modules.append(self)

        for name, module in self.modules:
            if isinstance(module, ModuleList):
                if recursive:
                    modules += module.get_modules(include, exclude, recursive)
            else:
                if self._check(name, module, include, exclude):
                    modules.append(module)

        return modules

    def module_control(self, name, module, mask_modules=(), apply_modules=(), start_module=None, end_module=None, **kwargs):
        """control the module is masked or applied,
        return `Ture` to mask module, `False` to apply module
        """
        # ignore `mask_modules` when `module.mask=True`
        if getattr(module, 'mask', False) or name in mask_modules:
            return True

        # ignore `apply_modules` when `module.apply=False`
        if not (getattr(module, 'apply', True) or name in apply_modules):
            return True

        mask_flag, _ = self._make_mask_flag(start_module, end_module, apply_modules)

        for i, (_name, module) in enumerate(self.modules):
            if _name == name and mask_flag[i]:
                return True

        return False

    def _make_mask_flag(self, start_module, end_module, apply_modules):
        start, end = 0, len(self.modules)
        is_shoot = [False, False]
        mask_flag = [True] * len(self.modules)
        for i, (_name, module) in enumerate(self.modules):
            _is_shoot = [False, False]
            if isinstance(module, ModuleList):
                _is_shoot = module._make_mask_flag(start_module, end_module, apply_modules)[1]

            if module.allow_start and _name == start_module or _is_shoot[0]:
                start = i
                is_shoot[0] = True

            if module.allow_end and _name == end_module or _is_shoot[1]:
                end = i + 1
                is_shoot[1] = True

            if _name in apply_modules:
                mask_flag[i] = False

        mask_flag[start:end] = [False] * (end - start)
        return mask_flag, is_shoot

    def apply_setting(self, obj, func_name, include=(), exclude=(), recursive=False):
        if self._check(self.name, self, include, exclude):
            setattr(self, func_name, obj)

        for name, module in self.modules:
            if isinstance(module, (Sequential, Pipeline)):
                if recursive:
                    module.apply_setting(obj, func_name, include, exclude, recursive)
            else:
                if self._check(name, module, include, exclude):
                    setattr(module, func_name, obj)

    def logger_(self, logger=None):
        self.apply_setting(log_utils.get_logger(logger), 'logger')

    def ignore_errors_(self, ignore=True):
        self.apply_setting(ignore, 'ignore_errors')

    def apply_(self, apply=True):
        self.apply_setting(apply, 'apply')

    def mask_(self, mask=True):
        self.apply_setting(mask, 'mask')

    def __getitem__(self, key):
        module = self.get_module(key)
        assert module is not None, f'"{key}" is not the module of {self.name}'
        return module

    def to_structure_dict(self):
        d = super().to_structure_dict()
        modules = []
        for name, module in self.modules:
            modules.append(module.to_structure_dict())

        d['modules'] = modules
        return d

    def to_structure_array(self):
        structure_array, cfgs = super().to_structure_array()
        modules = []
        for name, module in self.modules:
            structure_array_, cfgs_ = module.to_structure_array()
            modules.append(structure_array_)
            cfgs.update(cfgs_)
        structure_array = (structure_array, modules)
        return structure_array, cfgs


@base_module_tables.add_register()
class Pipeline(ModuleList):
    """run module step by step
    the next module start process until the last module has processed all the data,
    and the output of the last module will be the input of the next module
    """

    def on_process(self, obj, **kwargs):
        for name, module in self.modules:
            if self.module_control(name, module, **kwargs):
                continue

            _obj = module(obj, **kwargs)
            if isinstance(_obj, Exception):
                raise _obj

            if not self.inplace:
                obj = _obj

        return obj


@base_module_tables.add_register()
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


@base_module_tables.add_register()
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


@base_module_tables.add_register()
class IgnoreExceptionPipeline(Pipeline, RetryModule):
    """
    Usages:
        class E(Module):
            def on_process(self, obj, **kwargs):
                raise ValueError(obj)

        pipe = IgnoreExceptionPipeline(
            E()
        )
        pipe(0)
    """


@base_module_tables.add_register()
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

    def __init__(self, *args, fail_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_module = fail_module

    def on_process(self, obj, **kwargs):
        name = self.switch(obj, **kwargs)
        module = self.get_module(name)
        if module is None:
            if self.fail_module is None:
                raise ValueError(f'{name} is not found in {self.modules}')
            else:
                module = self.fail_module

        if self.module_control(name, module, **kwargs):
            return obj

        _obj = module(obj, **kwargs)

        if isinstance(_obj, Exception):
            raise _obj

        if not self.inplace:
            obj = _obj

        return obj

    def switch(self, obj, **kwargs) -> str | int:
        """return the name of selected module"""
        raise NotImplemented


@base_module_tables.add_register()
class SkipPipeline(Pipeline, SkipModule):
    """
    Usages:
        class A(Module):
            def on_process(self, obj, **kwargs):
                return obj

        pipe = SkipPipeline(
            A()
        )
        pipe(True)
    """


@base_module_tables.add_register()
class MultiProcessPipeline(Pipeline):
    """modules parallel by multiple processes
    each module will have the same inputs,
    use inplace mode to return the outputs"""
    n_pool = None
    inplace = True

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


@base_module_tables.add_register()
class MultiThreadPipeline(Pipeline):
    """modules parallel by multiple threads
    each module will have the same inputs,
    use inplace mode to return the outputs"""
    n_pool = None
    inplace = True

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


@base_module_tables.add_register()
class Sequential(ModuleList):
    """run data step by step
    the next data start process until the last data has been processed by all the modules,
    the first module must be an instance of `BaseSequentialInput`,
    if not provided, default creates a new `BaseSequentialInput` module
    """

    # only work with `ignore_iter_errors=True`
    skip_exception_return = False
    cache_all_results = True

    iter_callback_wrapper_ins = callbacks.CallbackWrapper
    iter_callback_wrapper_kwargs = dict()

    pbar_visualize = False

    def __init__(self, *modules, force_add_input=True, force_add_output=True, iter_success_callbacks=None, iter_failure_callbacks=None, **kwargs):
        if force_add_input and not isinstance(modules[0], BaseSequentialInput):
            modules = [BaseSequentialInput()] + list(modules)

        if force_add_output and not isinstance(modules[-1], BaseSequentialOutput):
            modules = list(modules) + [BaseSequentialOutput()]

        super().__init__(*modules, **kwargs)

        if iter_success_callbacks:
            self.iter_callback_wrapper_kwargs = configs.ConfigObjParse.merge_dict(self.iter_callback_wrapper_kwargs, dict(success_callbacks=iter_success_callbacks))

        if iter_failure_callbacks:
            self.iter_callback_wrapper_kwargs = configs.ConfigObjParse.merge_dict(self.iter_callback_wrapper_kwargs, dict(failure_callbacks=iter_failure_callbacks))

        # note, not necessary to use judgment statements, but for the convenience of debugging
        self.iter_callback_wrapper = callbacks.FakeCallbackWrapper()
        if self.iter_callback_wrapper_kwargs:
            self.add_iter_callback()

    def __call__(self, obj, **kwargs):
        # todo, `gen_kwargs` does not be wrapped in callback
        kwargs = configs.ConfigObjParse.merge_dict(kwargs, self.callback_wrapper.gen_kwargs(obj, **kwargs), inplace=True)
        kwargs = configs.ConfigObjParse.merge_dict(kwargs, self.iter_callback_wrapper.gen_kwargs(obj, **kwargs), inplace=True)
        kwargs = self.gen_kwargs(obj, **kwargs)
        return self._process(obj, **kwargs)

    def add_iter_callback(self):
        if isinstance(self.iter_callback_wrapper, callbacks.FakeCallbackWrapper):
            self.iter_callback_wrapper = self.iter_callback_wrapper_ins(module_name=self.name, **self.iter_callback_wrapper_kwargs)
            self._iter_result = self.iter_callback_wrapper.on_process_wrap(self._iter_result, sub_callback_step='on_iter')
            self._iter = self.iter_callback_wrapper.on_process_start_end_wrap(self._iter)

    @property
    def ignore_iter_errors(self):
        return self.iter_callback_wrapper.ignore_errors

    @ignore_iter_errors.setter
    def ignore_iter_errors(self, ignore):
        self.add_iter_callback()
        self.iter_callback_wrapper.ignore_errors = ignore

    def ignore_iter_errors_(self, ignore=True):
        self.apply_setting(ignore, 'ignore_iter_errors')

    def on_process(self, obj, **kwargs):
        return self._iter(obj, **kwargs)

    def _iter(self, obj, **kwargs):
        _, input_module = self.modules[0]
        _, output_module = self.modules[-1]
        results = []
        # todo, more elegant implementation?
        iter_objs, flag = self.iter_callback_wrapper.on_process(input_module, obj, return_exceptions_flag=True, sub_callback_step='on_iter_strat', **kwargs)
        if flag:
            return iter_objs
        if self.pbar_visualize:
            iter_objs = tqdm(iter_objs, desc=self.name)
        for iter_obj in iter_objs:
            iter_obj = self._iter_result(iter_obj, **kwargs)
            if self.skip_exception_return and isinstance(iter_obj, Exception):
                continue
            if self.cache_all_results:
                results.append(iter_obj)

        return self.iter_callback_wrapper.on_process(output_module, results, raw_obj=obj, sub_callback_step='on_iter_end', **kwargs)

    def _iter_result(self, iter_obj, **kwargs):
        return self._iter_module(iter_obj, **kwargs)

    def _iter_module(self, iter_obj, **kwargs):
        for name, module in self.modules[1:-1]:
            if self.module_control(name, module, **kwargs):
                continue

            _iter_obj = module(iter_obj, **kwargs)
            if isinstance(_iter_obj, Exception):
                raise _iter_obj

            if not self.inplace:
                iter_obj = _iter_obj
        return iter_obj


@base_module_tables.add_register()
class IterSequential(Sequential):
    """Each data result will yield out"""

    def add_callback(self):
        super().add_callback()
        self.logger.warning('There would be something wrong to use `callback_wrapper` in IterSequential. Using `iter_callback_wrapper` instead.')

    def add_iter_callback(self):
        if isinstance(self.iter_callback_wrapper, callbacks.FakeCallbackWrapper):
            self.iter_callback_wrapper = self.iter_callback_wrapper_ins(module_name=self.name, **self.iter_callback_wrapper_kwargs)
            self._iter_result = self.iter_callback_wrapper.on_process_wrap(self._iter_result, sub_callback_step='on_iter')
            self._iter = self.iter_callback_wrapper.on_process_start_wrap(self._iter)

    def _iter(self, obj, **kwargs):
        _, input_module = self.modules[0]
        _, output_module = self.modules[-1]
        results = []

        iter_objs, flag = self.iter_callback_wrapper.on_process(input_module, obj, return_exceptions_flag=True, sub_callback_step='on_iter_start', **kwargs)
        if flag:
            yield iter_objs
            return

        if self.pbar_visualize:
            iter_objs = tqdm(iter_objs, desc=self.name)
        for iter_obj in iter_objs:
            iter_obj = self._iter_result(iter_obj, **kwargs)
            if self.skip_exception_return and isinstance(iter_obj, Exception):
                continue
            if self.cache_all_results:
                results.append(iter_obj)

            yield iter_obj

        obj = self.iter_callback_wrapper.on_process(output_module, results, raw_obj=obj, sub_callback_step='on_iter_end', **kwargs)
        obj = self.iter_callback_wrapper.on_process_end(lambda obj, **kwargs: obj, obj, **kwargs)
        return obj


@base_module_tables.add_register()
class AsyncIterSequential(IterSequential):
    """must provide a input module, don't need output module"""
    def __init__(self, *modules, **kwargs):
        super().__init__(*modules, force_add_input=False, force_add_output=False, **kwargs)

    async def _iter(self, obj, **kwargs):
        _, input_module = self.modules[0]
        iter_objs, flag = self.iter_callback_wrapper.on_process(input_module, obj, return_exceptions_flag=True, sub_callback_step='on_iter_start', **kwargs)
        if flag:
            yield iter_objs
            return

        if self.pbar_visualize:
            iter_objs = asyncio.tqdm(iter_objs, desc=self.name)
        async for iter_obj in iter_objs:
            iter_obj = self._iter_result(iter_obj, **kwargs)
            if self.skip_exception_return and isinstance(iter_obj, Exception):
                continue

            yield iter_obj

        self.iter_callback_wrapper.on_process_end(lambda obj, sub_callback_step='on_iter_end', **kwargs: obj, obj, **kwargs)


@base_module_tables.add_register()
class LoopSequential(Sequential):
    """data will be used recurrently for each loop"""

    def _iter(self, obj, **kwargs):
        _, input_module = self.modules[0]
        _, output_module = self.modules[-1]

        iter_objs, flag = self.iter_callback_wrapper.on_process(input_module, obj, return_exceptions_flag=True, sub_callback_step='on_iter_strat', **kwargs)
        if flag:
            return iter_objs

        for iter_obj in iter_objs:
            # iter_obj will be merged in obj
            obj.update(iter_obj)
            obj = self._iter_result(obj, **kwargs)
            if self.skip_exception_return and isinstance(obj, Exception):
                continue

        return self.iter_callback_wrapper.on_process(output_module, obj, sub_callback_step='on_iter_end', **kwargs)


@base_module_tables.add_register()
class BatchSequential(Sequential):
    """run batch data step by step
    the next batch data start process until the last batch data has been processed by all the modules,
    and must have an input layer which returns an iterable obj
    """

    batch_size = 1

    def _iter(self, obj, mask_modules=(), **kwargs):
        _, input_module = self.modules[0]
        _, output_module = self.modules[-1]
        results = []
        i = 0
        iter_objs, flag = self.iter_callback_wrapper.on_process(input_module, obj, return_exceptions_flag=True, sub_callback_step='on_iter_strat', **kwargs)
        if flag:
            return iter_objs
        if self.pbar_visualize:
            iter_objs = tqdm(iter_objs, desc=self.name)
        batch_iter_obj = []
        for iter_obj in iter_objs:
            i += 1

            batch_iter_obj.append(iter_obj)
            if i < self.batch_size:
                continue

            _iter_objs = self._iter_result(batch_iter_obj, **kwargs)

            if not (self.skip_exception_return and isinstance(_iter_objs, Exception)) and self.cache_all_results:
                results += _iter_objs

            i = 0
            batch_iter_obj = []

        if batch_iter_obj:
            _iter_objs = self._iter_result(batch_iter_obj, **kwargs)
            if not (self.skip_exception_return and isinstance(_iter_objs, Exception)) and self.cache_all_results:
                results += _iter_objs

        return self.iter_callback_wrapper.on_process(output_module, results, raw_obj=obj, sub_callback_step='on_iter_end', **kwargs)


@base_module_tables.add_register()
class MultiProcessModuleSequential(Sequential):
    """modules parallel by multiple processes
    each module will have the same inputs
    """

    n_pool = None
    inplace = True

    def _iter_module(self, iter_obj, **kwargs):
        from multiprocessing.pool import Pool

        pool = Pool(self.n_pool)
        processes = {}
        for name, module in self.modules[1:-1]:
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


@base_module_tables.add_register()
class MultiProcessDataSequential(Sequential):
    """data parallel by multiple processes"""
    n_pool = None

    def _iter(self, obj, **kwargs):
        from multiprocessing.pool import Pool

        pool = Pool(self.n_pool)

        _, input_module = self.modules[0]
        _, output_module = self.modules[-1]
        processes = []
        results = []
        pbar = None
        iter_objs, flag = self.iter_callback_wrapper.on_process(input_module, obj, return_exceptions_flag=True, sub_callback_step='on_iter_strat', **kwargs)
        if flag:
            return iter_objs
        if self.pbar_visualize:
            # set a very small delay to avoid printing the pbar when initialization
            pbar = tqdm(desc=self.name, delay=1e-9)
        for iter_obj in iter_objs:
            processes.append(pool.apply_async(self._iter_module, args=(iter_obj,), kwds=kwargs))
            results.append(None)

            self.checkout_iter_results(processes, results, on_process=True, pbar=pbar, **kwargs)

        pool.close()
        pool.join()

        self.checkout_iter_results(processes, results, on_process=False, pbar=pbar, **kwargs)

        return self.iter_callback_wrapper.on_process(output_module, results, raw_obj=obj, sub_callback_step='on_iter_end', **kwargs)

    def checkout_iter_results(self, processes, results, on_process=True, pbar=None, **kwargs):
        for i, p in enumerate(processes):
            if p is None:
                continue

            # only when on_process, running the following scripts after the process is finished
            if on_process and not p._success:
                continue

            processes[i] = None
            iter_obj = self._iter_result(p, **kwargs)
            if self.skip_exception_return and isinstance(iter_obj, Exception):
                continue

            if self.cache_all_results:
                results[i] = iter_obj

            if self.pbar_visualize:
                pbar.update(1)

    def _iter_result(self, p, **kwargs):
        return p.get()


@base_module_tables.add_register()
class MultiThreadModuleSequential(Sequential):
    """module parallel by multiple threads,
    each module will have the same inputs,
    use inplace mode to return the outputs"""
    n_pool = None
    inplace = True

    def __init__(self, *modules, **kwargs):
        super().__init__(*modules, force_add_input=False, force_add_output=False, **kwargs)
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max_workers=self.n_pool)

    def _iter_module(self, iter_obj, **kwargs):
        threads = []
        for name, module in self.modules:
            if self.module_control(name, module, **kwargs):
                continue

            threads.append(self.pool.submit(module, iter_obj, **kwargs))

        for t in threads:
            _iter_obj = t.result()
            if isinstance(_iter_obj, Exception):
                raise _iter_obj

        # inplace mode to return the outputs
        return iter_obj


@base_module_tables.add_register()
class MultiThreadDataSequential(Sequential):
    """data parallel by multiple threads"""
    n_pool = None

    def __init__(self, *modules, **kwargs):
        super().__init__(*modules, **kwargs)
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max_workers=self.n_pool)

    def _iter(self, obj, **kwargs):
        _, input_module = self.modules[0]
        _, output_module = self.modules[-1]
        results = []
        threads = []
        pbar = None
        if self.pbar_visualize:
            # set a very small delay to avoid printing the pbar when initialization
            pbar = tqdm(desc=self.name, delay=1e-9)

        iter_objs, flag = self.iter_callback_wrapper.on_process(input_module, obj, return_exceptions_flag=True, sub_callback_step='on_iter_strat', **kwargs)
        if flag:
            return iter_objs

        for iter_obj in input_module(obj, **kwargs):
            threads.append(self.pool.submit(self._iter_module, iter_obj, **kwargs))
            results.append(None)

            # to avoid the threads accumulate too much, check each iter, and close the threads which is finished.
            self.checkout_iter_results(threads, results, on_process=True, pbar=pbar, **kwargs)

        self.checkout_iter_results(threads, results, on_process=False, pbar=pbar, **kwargs)
        results = [r for r in results if r is not None]
        return self.iter_callback_wrapper.on_process(output_module, results, raw_obj=obj, sub_callback_step='on_iter_end', **kwargs)

    def checkout_iter_results(self, threads, results, on_process=True, pbar=None, **kwargs):
        for i, t in enumerate(threads):
            if t is None:
                continue

            # only when on_process, running the following scripts after the thread is finished
            if on_process and t._state != 'FINISHED':
                continue

            threads[i] = None
            iter_obj = self._iter_result(t, **kwargs)
            if self.skip_exception_return and isinstance(iter_obj, Exception):
                continue

            if self.cache_all_results:
                results[i] = iter_obj

            if self.pbar_visualize:
                pbar.update(1)

    def _iter_result(self, t, **kwargs):
        return t.result()


@base_module_tables.add_register()
class BaseSequentialInput(Module):
    """default input module for Sequential
    do nothing, just return an iterable obj"""

    def on_process(self, objs: Optional[list], **kwargs):
        for obj in objs:
            yield obj


@base_module_tables.add_register()
class BaseSequentialOutput(Module):
    """default input module for Sequential
    do nothing, just return an iterable obj"""

    def on_process(self, objs: list, raw_obj=None, **kwargs):
        return objs


@base_module_tables.add_register()
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


@base_module_tables.add_register()
class BasePipelineInput(Module):
    """default input module for Sequential
    do nothing, just return the inputs obj"""

    def on_process(self, objs, *args, **kwargs):
        return objs


@base_module_tables.add_register()
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
