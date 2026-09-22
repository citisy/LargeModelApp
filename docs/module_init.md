# 用 Module.from_xxx() 初始化

`workflows.skeletons.Module` 可以用配置或结构描述把一棵模块树建出来。运行时数据约定、`Pipeline` / `Sequential` 的行为见 [Workflows](./workflows.md)。HTTP 里 `model_configs` 如何交到 `from_configs`，见 [Components](./components.md)。

```python
from workflows import skeletons
```

五种类方法：

| 方法 | 输入 | 适合 |
| --- | --- | --- |
| `from_configs` | 按类名索引的配置，外加已经建好的子模块 | 树写在代码里，只覆盖参数 |
| `from_structure_dict` | 嵌套 dict，每个节点自带 `config` | 同一类出现多次，且每次参数不同 |
| `from_structure_array` | 嵌套 tuple，外加一份扁平 `cfgs` | 结构短，同一类共用配置，并且要导出后再加载 |
| `from_structure_file` | json / yaml 文件 | 结构放在文件里，内容与 structure dict 相同 |
| `from_dify_config_file` | Dify workflow yaml | 从 Dify 导出直接得到可调用实例 |

---

## 1. 注册表

`from_structure_dict`、`from_structure_array`、`from_structure_file` 用节点上的 `name` 找类。`from_configs` 和 `from_dify_config_file` 不查这张表。

内置类登记在 `skeletons.base_module_tables`，键是类名，例如 `Pipeline`、`Sequential`、`SwitchPipeline`、`RetryModule`、`MultiThreadPipeline`、`BaseSequentialInput`、`KeepSequentialOutput`。与 [Workflows](./workflows.md) 里的类一一对应。`ModuleList` 没有登记，不能当作节点名。

业务类放进调用时传入的 `register_tables`。两种写法：

### dict

表里没有的名字会继续查内置表。结构里可以直接写 `Pipeline`、`Sequential`。

```python
reg = {
    'StepA': StepA,
    'StepB': StepB,
}
```

### RegisterTables

只查本表。结构里出现的每个名字都要先登记，包括要用到的内置类。

```python
from utils import op_utils

reg = op_utils.RegisterTables()

@reg.add_register()
class StepA(skeletons.Module):
    pass

reg.add_register()(skeletons.Pipeline)
reg.add_register()(skeletons.Sequential)
```

装饰器默认用类名当键。`@reg.add_register('别名')` 则用这个别名，结构里的 `name` 要写成 `别名`。

名字两边都没有时抛 `KeyError`。

---

## 2. from_configs

```python
Module.from_configs(cfgs, *modules, name=None, **kwargs) -> Module
```

步骤：

1. 决定配置键：参数 `name`，否则类属性 `name`，否则类名。
2. 取出 `cfgs[键]`，没有则用 `{}`。
3. 与 `kwargs` 深合并。同键以 `kwargs` 为准；两边都是 dict 时按键合并。
4. 合并结果写到**类属性** `config`。
5. 返回 `cls(*modules, cfgs=cfgs, name=键, **合并结果)`。

`Module.__init__` 会把这些关键字收进实例属性，因此类上的字段（如 `factor`、`retry_count`、`n_pool`）都可以被配置覆盖。

```python
from workflows import skeletons


class Scale(skeletons.Module):
    factor = 1

    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['value'] *= self.factor
        return obj


cfgs = {'Scale': {'factor': 10}}
print(Scale.from_configs(cfgs)({'value': 2}))
# {'value': 20}

# kwargs 覆盖 cfgs 里的同名字段
print(Scale.from_configs(cfgs, factor=3).factor)
# 3
```

嵌套 dict 会合并，缺的键保留：

```python
class Box(skeletons.Module):
    opt = {}


m = Box.from_configs({'Box': {'opt': {'a': 1, 'b': 1}}}, opt={'b': 2, 'c': 3})
print(m.opt)
# {'a': 1, 'b': 2, 'c': 3}
```

两边都是 list 时，按位置收成 `[原值, 新值]`。列表字段放在 `cfgs` 里即可，不要再用同名 `kwargs` 去覆盖。

`from_configs` 只配置**当前这次调用的类**。子模块要各自再调一次，容器在 `__init__` 里把同一份 `cfgs` 传下去：

```python
class StepA(skeletons.Module):
    flag = 'a'

    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['a'] = self.flag
        return obj


class StepB(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['b'] = True
        return obj


class Main(skeletons.Pipeline):
    def __init__(self, cfgs={}, **kwargs):
        super().__init__(
            StepA.from_configs(cfgs),
            StepB.from_configs(cfgs),
            **kwargs,
        )


cfgs = {'Main': {}, 'StepA': {'flag': 'A'}, 'StepB': {}}
print(Main.from_configs(cfgs)({}))
# {'a': 'A', 'b': True}
```

`components.api.create_app` 把 YAML 的 `model_configs` 原样交给 `Model.from_configs`。键用子类类名。任务类按上面的写法拆给子模块。见 [Components](./components.md) 第 7 节。

同一类连续调用时，类属性 `config` 保留**最后一次**合并结果。实例自己的字段仍是当次传入的值。`to_structure_dict` / `to_structure_array` 读的是类上的 `config`。

```python
a = Scale.from_configs({'Scale': {'factor': 2}})
b = Scale.from_configs({'Scale': {'factor': 9}})
print(a.factor, b.factor, a.config)
# 2 9 {'factor': 9}
```

需要同一类两份不同配置、并且导出结果互不覆盖时，做成两个子类，或改用 `from_structure_dict`（每个节点一份 `config`）。

`DictSequentialInput`、`ListPipelineInput` 的构造函数不接收 `cfgs` / `name`。这两类用 `from_structure_dict`，`config` 里只放它们声明过的参数（`var_keys` / `const_keys`、`select_keys`）。

---

## 3. from_structure_dict

```python
Module.from_structure_dict(structure_dict, register_tables) -> Module
```

一个节点：

```python
{
    'name': '注册名',
    'config': {},          # 构造参数，可省略
    'modules': [ ... ],    # 子节点，叶子可省略
}
```

按 `name` 取类，对每个子节点递归，然后 `类(*子实例, **config)`。`config` 里的键变成构造参数和实例属性，可以写 `name` 来改实例名（`mask_modules`、`get_module`、`start_module` 用这个名字）。

```python
from workflows import skeletons


class StepA(skeletons.Module):
    flag = 'a'

    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['a'] = self.flag
        return obj


class StepB(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['b'] = True
        return obj


reg = {'StepA': StepA, 'StepB': StepB}

structure = dict(
    name='Pipeline',
    config=dict(),
    modules=[
        dict(name='StepA', config=dict(flag='A')),
        dict(name='StepB', config=dict()),
    ],
)
m = skeletons.Module.from_structure_dict(structure, reg)
print(m({}))
# {'a': 'A', 'b': True}
```

`Pipeline` 不在 `reg` 里，由内置表补上。若 `register_tables` 是 `RegisterTables`，要先 `reg.add_register()(skeletons.Pipeline)`。

同一注册名可以出现多次，各自带自己的 `config`：

```python
class Scale(skeletons.Module):
    factor = 1

    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['value'] *= self.factor
        return obj


reg = {'Scale': Scale}
structure = dict(
    name='Pipeline',
    modules=[
        dict(name='Scale', config=dict(factor=2, name='S1')),
        dict(name='Scale', config=dict(factor=5, name='S2')),
    ],
)
m = skeletons.Module.from_structure_dict(structure, reg)
print([(name, mod.factor) for name, mod in m.modules])
# [('S1', 2), ('S2', 5)]
print(m({'value': 1}))
# {'value': 10}
```

### to_structure_dict

```python
module.to_structure_dict() -> dict
```

叶子返回 `{name, config}`。`ModuleList`（`Pipeline`、`Sequential` 等）再带上 `modules`。

`name` 是实例名。`config` 来自类属性 `config`，也就是 `from_configs` 写下的那份。`from_structure_dict` 只把 `config` 拆成实例属性，不写回 `self.config`，所以这条路径建出来的树，导出的 `config` 是空的。若节点 `config` 里改过 `name`，导出的名字是实例名（上例是 `S1`），和注册名 `Scale` 对不上，不能原样再加载。

要保存结构并再次加载，用下一节的 `from_structure_array` / `to_structure_array`。

### Sequential 会补输入输出

`Sequential` 默认在首尾插入 `BaseSequentialInput`、`BaseSequentialOutput`。首位已经是 `BaseSequentialInput` 时不再插入。`config` 里可以关：

```python
dict(
    name='Sequential',
    config=dict(force_add_input=False, force_add_output=False),
    modules=[dict(name='StepA', config=dict())],
)
```

自动补上的节点会出现在 `to_structure_dict()` 的 `modules` 里。

---

## 4. from_structure_array

```python
Module.from_structure_array(structure_array, register_tables, cfgs={}) -> Module
```

两种节点：

| 写法 | 含义 |
| --- | --- |
| `'StepA'` | 叶子 |
| `('Pipeline', ['StepA', 'StepB'])` | 容器和它的子节点 |

```python
from workflows import skeletons


class StepA(skeletons.Module):
    flag = 'a'

    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['a'] = self.flag
        return obj


class StepB(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['b'] = True
        return obj


class Main(skeletons.Pipeline):
    pass


reg = {'Main': Main, 'StepA': StepA, 'StepB': StepB}
structure_array = ('Main', ['StepA', ('Pipeline', ['StepB'])])
cfgs = dict(StepA=dict(flag='Q'))
m = skeletons.Module.from_structure_array(structure_array, reg, cfgs)
print(m({}))
# {'a': 'Q', 'b': True}
```

`Pipeline` 不在 `reg` 里，传入 `dict` 时走内置表。`Main`、`StepA`、`StepB` 要在表里。

每个节点内部调用 `from_configs`：

- 叶子：`类.from_configs(cfgs)`
- 容器：`类.from_configs(cfgs, *子模块)`

`cfgs` 的键是 `from_configs` 使用的配置键，一般是类名。结构里的字符串只负责查注册表。`@reg.add_register('别名')` 的别名和类名不一致时，`cfgs` 仍按类名写；也可以在类上设置 `name = '别名'`，让两边统一。

同一类的多个节点共用 `cfgs` 里的同一份配置。每个节点参数不同时，用子类区分，或改用 `from_structure_dict`。

源码文档字符串里的 `from_module_names` 是旧名字，调用用 `from_structure_array`。

### to_structure_array

```python
structure_array, cfgs = module.to_structure_array()
```

叶子返回 `(name, {name: config})`。容器把子节点收成 `(name, [子结构...])`，并合并各节点的 `cfgs`。

节点经过 `from_configs` 后，类属性 `config` 有值，导出的 `cfgs` 能对上。注册名和类名一致时，这对返回值可以再交给 `from_structure_array`。

同一类有多份实例时，导出的 `cfgs[类名]` 是该类最后一次 `from_configs` 的结果。

---

## 5. from_structure_file

```python
Module.from_structure_file(file_path, register_tables) -> Module
```

用 `os_lib.loader.auto_load` 按后缀读取，结果必须是 dict，然后走 `from_structure_dict`。结构文件用 json 或 yaml。

`flow.json`：

```json
{
  "name": "Pipeline",
  "config": {},
  "modules": [
    {"name": "StepA", "config": {"flag": "F"}},
    {"name": "StepB", "config": {}}
  ]
}
```

`reg` 与第 3 节相同（含 `StepA`、`StepB`）。

```python
m = skeletons.Module.from_structure_file('flow.json', reg)
print(m({}))
# {'a': 'F', 'b': True}
```

yaml 同样可以。jsonl、ini、txt 读出来不是这份结构 dict，不能当结构文件。

文件内容的约束与 `from_structure_dict` 相同，包括注册表、`Sequential` 自动补节点，以及 `to_structure_dict` 导出时 `config` 为空。

---

## 6. from_dify_config_file

```python
Module.from_dify_config_file(file_path) -> Module
```

读取 Dify 导出的 workflow YAML，由 `scripts.dfy2module.CodeGenerator` 生成 Python，在内存里执行，再从 `components.dify_helper.dify_register_modules` 取出该应用名下的 `Model` 并实例化。

返回值已经是 `BaseModelWithoutDb` 实例。调用方式和普通任务类一样，业务字段放在 `kwargs` 里：

```python
import os
from workflows import skeletons

os.environ['DIFY_MODEL_demo_model'] = 'your-model-id'

model = skeletons.Module.from_dify_config_file('dify工作流/test1.yml')
result = model({
    'task_id': 'demo',
    'kwargs': {},
})
```

生成类登记在分表里，表名是 YAML 的 `app.name`。串行步骤生成 `Pipeline`，并行生成 `MultiThreadPipeline`，分支生成 `SwitchPipeline`，迭代生成 `Sequential`。

LLM 节点的模型 id 在类体执行时读取环境变量，名字是 `DIFY_MODEL_` 加上模型名（非单词字符换成下划线）。环境变量要在调用 `from_dify_config_file` 之前设置。

这条路径不接收 `register_tables` 和 `cfgs`。子模块写在生成的 `Model.__init__` 里。要改结构或落盘时，用命令行：

```bash
python scripts/dify2module.py dify工作流/test1.yml -o dify_modules
```

---

## 7. config 里可以写什么

`config`（以及 `from_configs` 的 `kwargs`）里的键，会作为构造参数传给类。`Module` 及其多数子类用 `**kwargs` 收进实例属性，在 `on_process` 里通过 `self.xxx` 读取。自定义字段同样可以写。

| 字段 | 谁用 | 作用 |
| --- | --- | --- |
| `name` | `Module` | 实例名 |
| `retry_count` / `retry_wait` / `err_type` | `RetryModule`、`RetryPipeline` | 构造重试器，要在初始化时传入 |
| `n_pool` | `ThreadsLimitModule`、`MultiThreadPipeline`、`MultiThreadDataSequential` 等 | 线程或进程数；线程池类在 `__init__` 里读取 |
| `batch_size` | `BatchSequential`、`BatchSequentialInput` | 批大小 |
| `pbar_visualize` | `Sequential` | 是否显示进度条 |
| `cache_all_results` / `skip_exception_return` | `Sequential` | 是否缓存中间结果、是否丢掉失败项 |
| `force_add_input` / `force_add_output` | `Sequential.__init__` | 是否自动补输入、输出模块 |
| `inplace` / `mask` / `apply` / `allow_start` / `allow_end` | `Module` | 运行时控制，见 [Workflows](./workflows.md) |
| `var_keys` / `const_keys` | `DictSequentialInput` | 仅 `from_structure_dict` |
| `select_keys` | `ListPipelineInput` | 仅 `from_structure_dict` |
| `logger` | `Module.__init__` | 日志器；适合在 `from_configs(..., logger=logger)` 里传，不适合写进 json |

子模块放在 `modules`（或 `from_configs` 的位置参数）里，不放进 `config`。`SwitchPipeline` 的 `fail_module` 需要一个模块实例，结构文件里表达不了，建好树之后再赋：

```python
m.fail_module = StepB()
```

回调（`success_callbacks` 等）同样是对象，用 `from_configs` 的 `kwargs` 传入。

---

## 8. 怎么往上堆

固定代码结构、只换参数：任务类里调用 `from_configs`，配置来自 YAML `model_configs`。

结构也会变、每个节点参数不同：把树写成 dict 或 json/yaml，用 `from_structure_dict` / `from_structure_file`。注册表用 `dict`，内置容器不用重复登记。

结构短、还要 `to_structure_*` 再加载：用 `from_structure_array`。同一类多份不同参数时拆成子类。

Dify 导出的 workflow：用 `from_dify_config_file`。调用前设置 `DIFY_MODEL_*`。

| 需求 | 用 |
| --- | --- |
| 代码里写好树，配置覆盖字段 | `from_configs` |
| HTTP `model_configs` | `Model.from_configs`，子类在 `__init__` 里继续 `from_configs` |
| 节点各自一份参数 | `from_structure_dict` |
| 结构放文件 | `from_structure_file`（json / yaml） |
| 导出后再加载，同一类共用配置 | `from_structure_array` / `to_structure_array` |
| 直接跑 Dify workflow yaml | `from_dify_config_file` |
