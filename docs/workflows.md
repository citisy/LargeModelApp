# Workflows 编排框架

`workflows.skeletons` 用模块（Module）拼装数据处理流程。数据约定为可读写的 `dict`（下文称 `obj`），模块通过 `on_process(obj, **kwargs)` 读写字段后返回。

```python
from workflows import skeletons
```

调用方式：`result = module(obj, task_id='demo')`。`kwargs` 会沿流水线向下传，也可用于运行时裁剪（见文末）。

典型组合：外层 `Pipeline` 串阶段，中间 `Sequential` 扫列表，分支用 `SwitchPipeline` / `MultiThreadPipeline`。下面每个类都有独立示例，示例不依赖任何具体业务。

---

## 1. 基础模块

### Module

最小处理单元。生命周期：`on_process_start` → `on_process` → `on_process_end`。默认 `inplace=False`，以上一节点返回值作为下一节点输入。

```python
from workflows import skeletons


class Upper(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['text'] = obj['text'].upper()
        return obj


class AddLen(skeletons.Module):
    def on_process_start(self, obj, **kwargs):
        obj = dict(obj)
        obj['n'] = 0
        return obj

    def on_process(self, obj, **kwargs):
        obj['n'] = len(obj['text'])
        return obj


m = AddLen()
print(m({'text': 'hello'}))
# {'text': 'hello', 'n': 5}
```

常用属性：

| 属性 | 含义 |
| --- | --- |
| `name` | 模块名，默认类名；`mask_modules` / `get_module` 用这个名字 |
| `apply` / `mask` | 是否参与运行；`mask=True` 永久跳过 |
| `allow_start` / `allow_end` | 是否可作为 `start_module` / `end_module` 锚点 |
| `from_configs(cfgs)` | 用 `cfgs[name]` 覆盖构造参数 |

```python
class Scale(skeletons.Module):
    factor = 1

    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['value'] *= self.factor
        return obj


cfgs = {'Scale': {'factor': 10}}
print(Scale.from_configs(cfgs)({'value': 2}))
# {'value': 20}
```

`flow_chat('out', format='png')` 可导出结构图（需安装 Graphviz）。

### AsyncModule

异步版本，用 `await module(obj)`。

```python
import asyncio
from workflows import skeletons


class SleepDouble(skeletons.AsyncModule):
    async def on_process(self, obj, **kwargs):
        await asyncio.sleep(0.01)
        obj = dict(obj)
        obj['value'] *= 2
        return obj


async def main():
    print(await SleepDouble()({'value': 3}))


asyncio.run(main())
```

### ModuleList

容器基类（`Pipeline` / `Sequential` 的父类），本身不规定执行顺序。提供查找、替换、批量改属性。

```python
from workflows import skeletons


class A(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['a'] = 1
        return obj


class B(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['b'] = 2
        return obj


pipe = skeletons.Pipeline(A(), B())
print(pipe['A'].name)          # A
pipe.replace_module(B(name='A'))  # 按 name 替换
pipe.ignore_errors_(True)      # 递归设置（见 Pipeline）
```

---

## 2. 单模块控制

### RetryModule

失败后按 `retry_count` / `retry_wait` 重试。适合不稳定的外部调用。

```python
from workflows import skeletons


class Flaky(skeletons.RetryModule):
    retry_count = 3
    retry_wait = 0
    err_type = ValueError

    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['n'] = obj.get('n', 0) + 1
        if obj['n'] < 3:
            raise ValueError('not yet')
        return obj


print(Flaky()({}))
# {'n': 3}
```

`raise_type` 指定遇到该类异常时不再重试、直接抛出。

### IgnoreExceptionModule

捕获异常后走 `err_fn`，默认原样返回 `obj`。

```python
from workflows import skeletons


class MaybeFail(skeletons.IgnoreExceptionModule):
    err_type = ZeroDivisionError

    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['score'] = 1 / obj['x']
        return obj

    def err_fn(self, obj, **kwargs):
        obj = dict(obj)
        obj['score'] = None
        return obj


print(MaybeFail()({'x': 0}))
# {'x': 0, 'score': None}
```

### SkipModule

`skip()` 返回 `True` 时整段处理跳过。

```python
from workflows import skeletons


class OptionalUpper(skeletons.SkipModule):
    def skip(self, obj, **kwargs):
        return not obj.get('text')

    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['text'] = obj['text'].upper()
        return obj


print(OptionalUpper()({'text': ''}))
# {'text': ''}
print(OptionalUpper()({'text': 'ab'}))
# {'text': 'AB'}
```

### ThreadsLimitModule

用线程池把 `_process` 限制在 `n_pool` 路（常设为 1，给不可并发的资源排队）。

```python
from workflows import skeletons


class Exclusive(skeletons.ThreadsLimitModule):
    n_pool = 1

    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['done'] = True
        return obj


print(Exclusive()({'id': 1}))
```

### LoopModule

按 `check()` 循环；`gen_counter` / `update_counter` 维护计数。`check` 返回 `False` 时退出。

```python
from workflows import skeletons


class CountDown(skeletons.LoopModule):
    def gen_counter(self, obj, **kwargs):
        return obj['times']

    def check(self, obj, counter=None, **kwargs):
        return counter > 0

    def update_counter(self, obj, counter=0, **kwargs):
        return counter - 1

    def on_process(self, obj, counter=None, **kwargs):
        obj = dict(obj)
        obj.setdefault('trace', []).append(counter)
        return obj


print(CountDown()({'times': 3}))
# {'times': 3, 'trace': [3, 2, 1]}
```

`check_before_loop=False` 时先跑再判断（至少执行一次）。

---

## 3. Pipeline：按模块串行 / 分支 / 并行

`Pipeline` 按注册顺序执行子模块，上一步输出作为下一步输入。

```python
from workflows import skeletons


class Trim(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['text'] = obj['text'].strip()
        return obj


class Upper(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['text'] = obj['text'].upper()
        return obj


pipe = skeletons.Pipeline(Trim(), Upper())
print(pipe({'text': '  hi  '}))
# {'text': 'HI'}
```

运行时跳过某步：

```python
pipe({'text': '  hi  '}, mask_modules=['Upper'])
# {'text': 'hi'}
```

给需要当锚点的子模块设 `allow_start=True` / `allow_end=True` 后，可用 `start_module` / `end_module` 只跑一段。

### LoopPipeline

`Pipeline` + `LoopModule`：每一轮把整条流水线再跑一遍。

```python
from workflows import skeletons


class Tick(skeletons.Module):
    def on_process(self, obj, counter=None, **kwargs):
        obj = dict(obj)
        obj['value'] += 1
        return obj


class Until(skeletons.LoopPipeline):
    def check(self, obj, counter=None, **kwargs):
        return obj['value'] < 5


print(Until(Tick())({'value': 0}))
# {'value': 5}
```

### RetryPipeline

整条流水线作为一次可重试单元。

```python
from workflows import skeletons


class IncThenFail(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['n'] = obj.get('n', 0) + 1
        if obj['n'] < 2:
            raise RuntimeError('retry')
        return obj


pipe = skeletons.RetryPipeline(IncThenFail(), retry_count=3, retry_wait=0, err_type=RuntimeError)
print(pipe({}))
```

### IgnoreExceptionPipeline

流水线任一步抛错时吞掉并返回当前 `obj`（或 `err_fn` 的结果）。

```python
from workflows import skeletons


class Boom(skeletons.Module):
    def on_process(self, obj, **kwargs):
        raise RuntimeError('boom')


print(skeletons.IgnoreExceptionPipeline(Boom())({'ok': True}))
# {'ok': True}
```

### SwitchPipeline

实现 `switch()`，返回子模块的 `name` 或下标，只跑选中的那一个。找不到时走 `fail_module`。

```python
from workflows import skeletons


class PathA(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['out'] = 'A'
        return obj


class PathB(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['out'] = 'B'
        return obj


class ByKind(skeletons.SwitchPipeline):
    def switch(self, obj, **kwargs):
        return 0 if obj['kind'] == 'a' else 1


pipe = ByKind(PathA(), PathB())
print(pipe({'kind': 'b'}))
# {'kind': 'b', 'out': 'B'}
```

### SkipPipeline

`skip()` 为真则整条流水线不执行。

```python
from workflows import skeletons


class Work(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['done'] = True
        return obj


class IfReady(skeletons.SkipPipeline):
    def skip(self, obj, **kwargs):
        return not obj.get('ready')


print(IfReady(Work())({'ready': False}))
# {'ready': False}
print(IfReady(Work())({'ready': True}))
# {'ready': True, 'done': True}
```

### MultiThreadPipeline

子模块**并行**，共享同一个 `obj`（`inplace=True`）。各子模块必须原地 `obj.update(...)` 再 `return obj`，不要返回全新 dict，否则结果不会合并。适合互不依赖、写不同字段的分支。

```python
from workflows import skeletons


class AddX(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj['x'] = obj['n'] + 1
        return obj


class AddY(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj['y'] = obj['n'] * 2
        return obj


pipe = skeletons.MultiThreadPipeline(AddX(), AddY())
print(pipe({'n': 3}))
# {'n': 3, 'x': 4, 'y': 6}
```

子模块异常会在 `t.result()` 时抛出。

### MultiProcessPipeline

多进程并行，同样 `inplace=True`。子进程里对 `obj` 的修改**不会**自动写回主进程；`apply_async` 的返回值目前只用于检查异常。适合写文件、调外部服务等副作用任务。需要汇总结果时用 `MultiThreadPipeline`，或自行实现合并。

```python
from workflows import skeletons


class SideEffect(skeletons.Module):
    def on_process(self, obj, **kwargs):
        # 可在此写磁盘 / 发请求
        return obj


if __name__ == '__main__':
    skeletons.MultiProcessPipeline(SideEffect(), SideEffect())({'id': 1})
```

类需定义在可 pickle 的模块顶层，并用 `if __name__ == '__main__'` 启动。

---

## 4. Sequential：按数据迭代

`Sequential` 先用**输入模块**把 `obj` 拆成若干 `iter_obj`，每个元素依次经过中间模块，最后由**输出模块**聚合。

未指定时会自动包上 `BaseSequentialInput` 和 `BaseSequentialOutput`：输入假定 `obj` 可迭代，输出返回结果列表。

```python
from workflows import skeletons


class Square(skeletons.Module):
    def on_process(self, obj, **kwargs):
        return obj * obj


seq = skeletons.Sequential(Square())
print(seq([1, 2, 3, 4]))
# [1, 4, 9, 16]
```

`pbar_visualize=True` 显示进度条。`cache_all_results=False` 不缓存中间列表（省内存，输出模块拿到的 `objs` 可能为空）。`ignore_iter_errors=True` 时单条失败可继续；再开 `skip_exception_return=True` 会丢掉失败项。

### BaseSequentialInput

默认：对可迭代 `obj` 逐项 `yield`。自定义时按条构造子任务（把父字段拷进每条是常见写法）。

```python
from workflows import skeletons


class ItemInput(skeletons.BaseSequentialInput):
    def on_process(self, obj, **kwargs):
        for i, item in enumerate(obj['items']):
            yield {'item': item, 'index': i, 'prefix': obj['prefix']}


class Join(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['text'] = f"{obj['prefix']}{obj['item']}"
        return obj


seq = skeletons.Sequential(ItemInput(), Join())
print(seq({'prefix': '#', 'items': ['a', 'b']}))
```

### BaseSequentialOutput

默认返回结果列表。自定义可写回 `raw_obj`（原始输入）。

```python
from workflows import skeletons


class Collect(skeletons.BaseSequentialOutput):
    def on_process(self, objs, raw_obj=None, **kwargs):
        raw_obj = dict(raw_obj)
        raw_obj['texts'] = [o['text'] for o in objs]
        return raw_obj


seq = skeletons.Sequential(ItemInput(), Join(), Collect())
print(seq({'prefix': '#', 'items': ['a', 'b']}))
# {'prefix': '#', 'items': ['a', 'b'], 'texts': ['#a', '#b']}
```

### KeepSequentialOutput

忽略迭代结果列表，返回进入 Sequential 时的 `raw_obj`。中间模块要改父对象请写 `raw_obj`。

```python
from workflows import skeletons


class ItemInput(skeletons.BaseSequentialInput):
    def on_process(self, obj, **kwargs):
        for item in obj['items']:
            yield {'item': item}


class Acc(skeletons.Module):
    def on_process(self, obj, raw_obj=None, **kwargs):
        raw_obj.setdefault('seen', []).append(obj['item'])
        return obj


seq = skeletons.Sequential(
    ItemInput(),
    Acc(),
    skeletons.KeepSequentialOutput(),
)
print(seq({'items': [10, 20]}))
# {'items': [10, 20], 'seen': [10, 20]}
```

### DictSequentialInput

把「若干等长列表 + 若干常量」拆成多条 dict。

```python
from workflows import skeletons


class SumPair(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['s'] = obj['xs'] + obj['ys']
        return obj


seq = skeletons.Sequential(
    skeletons.DictSequentialInput(var_keys=['xs', 'ys'], const_keys=['bias']),
    SumPair(),
)
print(seq({'xs': [1, 2], 'ys': [10, 20], 'bias': 0}))
# [{'xs': 1, 'ys': 10, 'bias': 0, 's': 11}, ...]
```

### BatchSequentialInput

在输入侧按 `batch_size` 聚批后再 `yield` 一个 list。

```python
from workflows import skeletons


class BatchLen(skeletons.Module):
    def on_process(self, obj, **kwargs):
        return len(obj)


seq = skeletons.Sequential(
    skeletons.BatchSequentialInput(batch_size=2),
    BatchLen(),
)
print(seq([1, 2, 3, 4, 5]))
# [2, 2, 1]
```

注意：`Sequential` 若再自动包一层默认 Input，可能套两层。需要精确控制时，把自定义 Input 放在第一位（框架发现首位已是 `BaseSequentialInput` 就不会再插）。

### IterSequential

每处理完一条就 `yield`，适合流式消费。

```python
from workflows import skeletons


class Square(skeletons.Module):
    def on_process(self, obj, **kwargs):
        return obj * obj


for x in skeletons.IterSequential(Square())([1, 2, 3]):
    print(x)
```

不建议在 `IterSequential` 上挂普通 `callback_wrapper`，用 `iter_success_callbacks` / `iter_failure_callbacks`。

### AsyncIterSequential

异步迭代输入。需要自己提供输入模块（不会自动加 Input/Output）。

```python
import asyncio
from workflows import skeletons


class AsyncItems(skeletons.AsyncModule):
    async def on_process(self, obj, **kwargs):
        for x in obj:
            yield x


class Plus(skeletons.Module):
    def on_process(self, obj, **kwargs):
        return obj + 1


async def main():
    seq = skeletons.AsyncIterSequential(AsyncItems(), Plus())
    async for x in seq([1, 2, 3]):
        print(x)


asyncio.run(main())
```

普通同步生成器可包 `async_sequential_input_wrapper` 再给异步 Sequential 用。

### LoopSequential

每一轮把 `iter_obj` **merge 进同一个 `obj`**（`obj.update(iter_obj)`），再跑中间模块。适合「逐步往同一份状态里填字段」。

```python
from workflows import skeletons


class FieldInput(skeletons.BaseSequentialInput):
    def on_process(self, obj, **kwargs):
        for k, v in obj['fields']:
            yield {k: v}


class Nop(skeletons.Module):
    def on_process(self, obj, **kwargs):
        return obj


seq = skeletons.LoopSequential(FieldInput(), Nop())
print(seq({'fields': [('a', 1), ('b', 2)], 'keep': True}))
```

输出模块收到的是合并后的 `obj`，不是列表。

### BatchSequential

在 Sequential 迭代器上按 `batch_size` 聚批，再把**一批**交给中间模块（中间模块的 `obj` 是 list）。

```python
from workflows import skeletons


class SumBatch(skeletons.Module):
    def on_process(self, obj, **kwargs):
        return sum(obj)


seq = skeletons.BatchSequential(SumBatch(), batch_size=2)
print(seq([1, 2, 3, 4, 5]))
# [3, 7, 5]
```

### MultiThreadDataSequential

数据并行：多条 `iter_obj` 同时跑同一套中间模块。`n_pool` 控制线程数。

```python
from workflows import skeletons


class SlowSquare(skeletons.Module):
    def on_process(self, obj, **kwargs):
        return {'src': obj, 'dst': obj * obj}


class ToList(skeletons.BaseSequentialOutput):
    def on_process(self, objs, raw_obj=None, **kwargs):
        return [o['dst'] for o in objs if o]


seq = skeletons.MultiThreadDataSequential(
    SlowSquare(),
    ToList(),
    n_pool=4,
)
print(seq([1, 2, 3, 4]))
```

单条失败且未开 `ignore_iter_errors` 时会打断。长列表分片处理时常用这个。

### MultiProcessDataSequential

与上类似，但是多进程。子任务必须可 pickle，入口加 `if __name__ == '__main__'`。

```python
from workflows import skeletons


class Square(skeletons.Module):
    def on_process(self, obj, **kwargs):
        return obj * obj


if __name__ == '__main__':
    seq = skeletons.MultiProcessDataSequential(Square(), n_pool=2)
    print(seq([1, 2, 3, 4]))
```

### MultiThreadModuleSequential

对**同一条** `iter_obj`，多个中间模块线程并行。默认 `inplace=True`，模块应原地改 `iter_obj`。构造时 `force_add_input=False`（类已关闭自动 Input/Output），需要自己保证输入可迭代或放在外层 Sequential 里。

```python
from workflows import skeletons


class SetA(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj['a'] = 1
        return obj


class SetB(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj['b'] = 2
        return obj


# 作为 Pipeline 的一步、处理单条 dict 时：
class OneItemInput(skeletons.BaseSequentialInput):
    def on_process(self, obj, **kwargs):
        yield obj


seq = skeletons.Sequential(
    OneItemInput(),
    skeletons.MultiThreadModuleSequential(SetA(), SetB(), n_pool=2),
    skeletons.KeepSequentialOutput(),
)
print(seq({'n': 0}))
```

### MultiProcessModuleSequential

同一条数据上多模块多进程并行，必须实现 `merge_outputs(results: dict)`。

```python
from workflows import skeletons


class Left(skeletons.Module):
    def on_process(self, obj, **kwargs):
        return {'left': obj['n'] - 1}


class Right(skeletons.Module):
    def on_process(self, obj, **kwargs):
        return {'right': obj['n'] + 1}


class MergeBoth(skeletons.MultiProcessModuleSequential):
    def merge_outputs(self, results, **kwargs):
        out = {}
        for part in results.values():
            out.update(part)
        return out


if __name__ == '__main__':
    seq = MergeBoth(Left(), Right(), n_pool=2)
    print(seq([{'n': 10}]))
```

---

## 5. Pipeline 输入辅助

### BasePipelineInput

原样返回输入，占位用。

```python
from workflows import skeletons

print(skeletons.BasePipelineInput()({'k': 1}))
# {'k': 1}
```

### ListPipelineInput

把 `List[dict]` 收成 `Dict[str, list]`。出现在 `select_keys` 里的键按条 append，其余键保留最后一次（或按实现写成标量覆盖）。

```python
from workflows import skeletons

inp = skeletons.ListPipelineInput(select_keys=['id', 'score'])
print(inp([
    {'id': 1, 'score': 0.1, 'src': 'x'},
    {'id': 2, 'score': 0.2, 'src': 'x'},
]))
```

常放在 `Sequential` 之后、`Pipeline` 下一步之前，把「逐条结果」折回列式结构。

---

## 6. 用配置 / 结构装配

参数、注册表、导出后再加载，以及 Dify 文件入口，见 [用 Module.from_xxx() 初始化](./module_init.md)。

### from_structure_dict

```python
from workflows import skeletons
from utils import op_utils

reg = op_utils.RegisterTables()


@reg.add_register()
class Main(skeletons.Pipeline):
    pass


@reg.add_register()
class StepA(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['a'] = True
        return obj


@reg.add_register()
class StepB(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['b'] = True
        return obj


structure = dict(
    name='Main',
    config=dict(),
    modules=[
        dict(name='StepA', config=dict()),
        dict(name='StepB', config=dict()),
    ],
)
m = skeletons.Module.from_structure_dict(structure, reg)
print(m({}))
# {'a': True, 'b': True}
print(m.to_structure_dict())
```

上面的 `reg` 是 `RegisterTables`，结构里出现的每个名字都要先登记。传入 `dict` 时，表里没有的名字才会去内置表（`Pipeline`、`Sequential` 等）查找。

### from_structure_array

用嵌套元组描述树：`(容器名, [子节点...])`。

```python
structure_array = ('Main', ['StepA', ('Main', ['StepA']), 'StepB'])
cfgs = dict(Main=dict(), StepA=dict(), StepB=dict())
m = skeletons.Module.from_structure_array(structure_array, reg, cfgs)
```

### from_structure_file

文件解析成 dict 后走 `from_structure_dict`（支持 `os_lib.loader.auto_load` 能读的格式）。

```python
m = skeletons.Module.from_structure_file('flow.json', reg)
```

---

## 7. 运行时裁剪与回调

调用任意 `ModuleList` 时可传：

| 参数 | 作用 |
| --- | --- |
| `mask_modules=['Name']` | 跳过这些名字 |
| `apply_modules=['Name']` | 白名单（与 start/end 一起决定窗口） |
| `start_module='Name'` | 从该 `allow_start` 模块开始 |
| `end_module='Name'` | 到该 `allow_end` 模块结束（含） |

```python
class A(skeletons.Module):
    allow_start = True
    allow_end = True

    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['a'] = 1
        return obj


class B(skeletons.Module):
    allow_start = True
    allow_end = True

    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['b'] = 1
        return obj


pipe = skeletons.Pipeline(A(), B())
print(pipe({}, start_module='B'))
# {'b': 1}
```

`success_callbacks` / `failure_callbacks` 传给 `CallbackWrapper`。`ignore_errors = True` 时失败不中断（需已挂 callback）。Sequential 另有 `iter_success_callbacks` 和 `ignore_iter_errors`。

---

## 8. 怎么往上堆

下面是一个「外层串行 + 列表分发 + 分支 + 并行字段」的骨架，对应真实项目里最常见的搭法，但不绑定任何业务字段含义。

```python
from workflows import skeletons


class Load(skeletons.RetryModule):
    retry_count = 2
    retry_wait = 0

    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj.setdefault('items', ['x', 'y'])
        return obj


class ItemInput(skeletons.BaseSequentialInput):
    def on_process(self, obj, **kwargs):
        for item in obj['items']:
            yield {'item': item}


class Branch(skeletons.SwitchPipeline):
    def switch(self, obj, **kwargs):
        return 0 if obj['item'] == 'x' else 1


class Left(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['tag'] = 'L'
        return obj


class Right(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['tag'] = 'R'
        return obj


class ScoreA(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj['sa'] = 1
        return obj


class ScoreB(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj['sb'] = 2
        return obj


class Collect(skeletons.BaseSequentialOutput):
    def on_process(self, objs, raw_obj=None, **kwargs):
        raw_obj = dict(raw_obj)
        raw_obj['rows'] = objs
        return raw_obj


flow = skeletons.Pipeline(
    Load(),
    skeletons.Sequential(
        ItemInput(),
        Branch(Left(), Right()),
        skeletons.MultiThreadPipeline(ScoreA(), ScoreB()),
        Collect(),
    ),
)

print(flow({}))
```

选择建议：

| 需求 | 用 |
| --- | --- |
| 固定阶段 A→B→C | `Pipeline` |
| 对列表逐条相同处理 | `Sequential` |
| 条与条互相独立、要加速 | `MultiThreadDataSequential` |
| 同一条上互不依赖的字段 | `MultiThreadPipeline`（原地写 `obj`） |
| 按字段走不同子图 | `SwitchPipeline` |
| 条件不满足整段不跑 | `SkipModule` / `SkipPipeline` |
| 外部调用不稳 | `RetryModule` / `RetryPipeline` |
| 重复直到收敛 | `LoopModule` / `LoopPipeline` |
