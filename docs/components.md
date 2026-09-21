# Components 组件

`components` 建在 `workflows.skeletons` 之上，提供任务入口、落库回调、HTTP 装配，以及若干外部服务客户端。数据约定与编排框架相同：`obj` 为 `dict`。

典型请求体：

```python
{
    'task_id': 'demo',
    'kwargs': {'text': 'hello'}  # 业务字段放这里
}
```

任务类会先取出 `kwargs`，跑完流水线后再包成 `{task_id, data}`。

```python
from components import base, template, db, api
from components.sdks import openai
```

下面每个类都有独立示例，不绑定任何具体业务场景。

---

## 1. 请求 / 响应模板 `template`

### BaseRequest

HTTP 入参基类。默认生成 8 位 `task_id`。

```python
from components.template import BaseRequest
from pydantic import BaseModel


class ReqData(BaseModel):
    text: str = ''


class Request(BaseRequest):
    kwargs: ReqData


req = Request(kwargs=ReqData(text='hello'))
print(req.task_id, req.dict())
```

可按需自行扩展 `callback_url`、`mask_modules` 等字段（模板里这些项默认注释掉了）。

### BaseSuccessResponse / BaseErrResponse

```python
from components.template import BaseSuccessResponse, BaseErrResponse

ok = BaseSuccessResponse(task_id='demo', data={'n': 1})
err = BaseErrResponse(task_id='demo', code=500, message='boom')
print(ok.dict())
print(err.dict())
```

成功默认 `code=200`。日志流水线把未捕获异常转成 `BaseErrResponse`（见 `BaseLogPipeline`）。

---

## 2. 任务基类 `base`

### BaseTaskPipeline

从完整请求里抽出 `kwargs` 作为流水线输入，结束时再包回。

```python
from components.base import BaseTaskPipeline
from workflows import skeletons


class Echo(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['echo'] = obj.get('text', '')
        return obj


class App(BaseTaskPipeline):
    pass


pipe = App(Echo())
print(pipe({'task_id': 't1', 'kwargs': {'text': 'hi'}}, task_id='t1'))
# {'task_id': 't1', 'data': {'text': 'hi', 'echo': 'hi'}}
```

`gen_kwargs` 会把请求顶层除 `kwargs` 外的字段（如 `task_id`）传给后续节点。

### BaseLogPipeline

默认挂耗时日志、标准错误回调，并 `ignore_errors=True`。异常会变成错误响应 dict，而不是把进程打崩。

```python
from components.base import BaseLogPipeline
from workflows import skeletons


class Boom(skeletons.Module):
    def on_process(self, obj, **kwargs):
        raise ValueError('oops')


class App(BaseLogPipeline):
    pass


print(App(Boom())({'x': 1}, task_id='t1'))
# {'task_id': 't1', 'code': 500, 'message': 'ValueError: oops', 'data': {}}
```

框架内 `workflows.exceptions._BaseException` 会带上自己的 `code` / `message`。

### BaseModelWithoutDb

`BaseLogPipeline` + `BaseTaskPipeline`，无数据库。大多数无状态接口继承它。

```python
from components.base import BaseModelWithoutDb
from workflows import skeletons


class Upper(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['text'] = obj['text'].upper()
        return obj


class Model(BaseModelWithoutDb):
    pass


m = Model(Upper())
print(m({'task_id': 't1', 'kwargs': {'text': 'ab'}}, task_id='t1'))
# {'task_id': 't1', 'data': {'text': 'AB'}}
```

### BaseModelWithMysqlDb

在无库版本上混入 `MysqlDbModule`。`query_data_first=True`（默认）时：按 `task_id` 查一行；没有则插入仅含 `task_id` 的记录；再把库字段与本次 `kwargs` 合并后进入流水线。适合「同一 task 可续跑 / 可查询」的任务。

```python
from components.base import BaseModelWithMysqlDb
from workflows import skeletons


class Work(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['done'] = True
        return obj


class Model(BaseModelWithMysqlDb):
    mysql_table = 'demo_task'


# 需配置环境变量 mysql_host / mysql_port / mysql_user / mysql_pwd / mysql_db
# m = Model(Work())
# m({'task_id': 't1', 'kwargs': {'text': 'x'}}, task_id='t1')
```

基类**不会**自动把失败写入数据库。需要失败落库时，在子模块上用 `MysqlCallbackModule`，或自行 `register_failure_callback`。

### AsyncBaseTaskSequential / AsyncBaseLogSequential / AsyncBaseModel

异步流式入口。`on_process_start` 同样取出 `kwargs`。日志版把异常序列化成 JSON 字符串，方便 `StreamingResponse` 写出。

```python
import asyncio
from components.base import AsyncBaseModel
from workflows import skeletons


class Chunks(skeletons.AsyncModule):
    async def on_process(self, obj, **kwargs):
        text = obj.get('text', '')
        for ch in text:
            yield ch


class Model(AsyncBaseModel):
    pass


async def main():
    m = Model(Chunks())
    async for part in m({'task_id': 't1', 'kwargs': {'text': 'ab'}}, task_id='t1'):
        print(part)


# asyncio.run(main())
```

需自己提供异步输入模块（见 [Workflows](./workflows.md) 的 `AsyncIterSequential`）。

### BaseCallbackModule

给模块挂上 `CallbackWrapper`，并通过 `global_cacher_keys` 把实例属性塞进 `general_kwargs['global_caches']`，供成功/失败回调写入。

```python
from components.base import BaseCallbackModule


class Step(BaseCallbackModule):
    global_cacher_keys = ['tag']
    tag = 'v1'

    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['ok'] = True
        return obj
```

### MysqlCallbackModule

成功 / 失败时更新 MySQL。常用类属性：

| 属性 | 含义 |
| --- | --- |
| `mysql_table` | 表名 |
| `mysql_cacher_keys` | 从 `obj` 里取出写入的字段 |
| `mysql_filter_mapping` | `{库字段: obj字段}`，用作更新条件，默认 `id → id` |
| `mysql_status_code_key` | 写入成功码 / 失败码的列 |
| `mysql_error_key` | 失败信息列，默认 `error_msg` |
| `success_code` / `error_code` | 默认 100 / 500 |
| `allow_duplicates` | 是否允许插入重复行 |

```python
from components.base import MysqlCallbackModule


class Persist(MysqlCallbackModule):
    mysql_table = 'demo_task'
    mysql_cacher_keys = ['result']
    mysql_filter_mapping = {'task_id': 'task_id'}
    mysql_status_code_key = 'status'

    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['result'] = obj.get('text', '').upper()
        return obj
```

关回调：`add_mysql_callback = False`。

### UrlCallbackModule

请求里带 `callback_url` 时，结束后 POST 一份成功或失败 JSON。默认只挂失败回调；成功需 `add_url_success_callback = True`。

```python
from components.base import UrlCallbackModule


class Notify(UrlCallbackModule):
    add_url_success_callback = True
    add_url_fail_callback = True
    url_cacher_keys = ['result']

    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['result'] = 1
        return obj


# Notify()(
#     {'result': 1},
#     task_id='t1',
#     callback_url='http://127.0.0.1:8000/hook',
# )
```

### TempMysqlCallbackModule

在正式表之外，再往 `temp_mysql_table` 写一份成功快照（例如模型调用明细）。用 `temp_mysql_cacher_keys` / `temp_mysql_global_cacher_keys` 控制写入字段。

```python
from components.base import TempMysqlCallbackModule


class Step(TempMysqlCallbackModule):
    mysql_table = 'demo_task'
    temp_mysql_table = 'demo_task_temp'
    mysql_cacher_keys = ['result']
    temp_mysql_cacher_keys = ['prompt', 'content']
```

---

## 3. 数据库 `db`

连接信息读环境变量：`mysql_host` / `mysql_port` / `mysql_user` / `mysql_pwd` / `mysql_db`。`get_one` / `cache_one` 对 `OperationalError` 默认重试 3 次。

### MysqlDbModule

只提供 `self.cacher`，本身不做读写。

```python
from components.db import MysqlDbModule

mod = MysqlDbModule(mysql_table='demo_task')
# row = mod.cacher.get_one(task_id='t1', convert_to_json=True)
# mod.cacher.cache_one({'task_id': 't1', 'status': 200}, task_id='t1', allow_duplicates=False)
```

### UpdateMysqlDbModule

在 `on_process_end` 把指定字段写回。`filter_mapping` 决定更新哪一行。

```python
from components.db import UpdateMysqlDbModule


class Save(UpdateMysqlDbModule):
    mysql_table = 'demo_task'
    cacher_keys = ['result', 'status']
    filter_mapping = {'task_id': 'task_id'}


# 放在 Pipeline 末尾：上游把 result 写入 obj 即可落库
```

### FakeCacher

空实现，表名为空时回调会用它，避免测试环境连库。

```python
from components.db import FakeCacher

c = FakeCacher()
assert c.get_one() == {}
```

### MysqlDbSuccessCallback / MySqlDbErrCallback

给 `CallbackWrapper` 用的落库回调。一般不要手动调，由 `MysqlCallbackModule` 注册。成功时按 `cacher_keys` 写字段并清空 `error_key`；失败时写入 `error_msg` 和状态码。

---

## 4. 回调 `_callbacks`

### TimeLoggerCallback

成功结束打一条带 `task_id` 的耗时日志。`BaseLogPipeline` 默认已挂。

### TimeDbCacheCallback

把耗时写进表的某一列。

```python
from components._callbacks import TimeDbCacheCallback
from workflows import skeletons


class Work(skeletons.Module):
    def on_process(self, obj, **kwargs):
        return obj


m = Work(success_callbacks=[
    TimeDbCacheCallback(
        mysql_table='demo_task',
        cache_key='duration',
        db_filter_mapping={'task_id': 'task_id'},
    )
])
```

### StdErrCallback

失败时打 traceback。`BaseLogPipeline` 默认已挂。

### UrlSuccessCallback / UrlErrCallback

见 `UrlCallbackModule`。成功体用 `BaseSuccessResponse`，失败体用 `BaseErrResponse`。没有 `callback_url` 则什么都不发。

---

## 5. HTTP 装配 `api`

### create_app

读一份 YAML/dict，实例化模型并挂到 FastAPI。

```yaml
Global:
  host: 0.0.0.0
  port: 8000

Log: {}

Api:
  /demo:
    /docs:
      title: Demo
      version: 1.0.0
      app_func: components.api.add_docs

    /echo:
      apply: true
      add_sync: true
      add_async: true
      num_async_worker: 2
      model_instance: mypkg.models.echo.Model
      request_template: mypkg.templates.echo.Request
      response_template: mypkg.templates.echo.Response
      method_configs:
        summary: echo
      model_configs:
        SomeChild:
          timeout: 10
```

```python
from components.api import create_app
from utils import os_lib

config = os_lib.loader.load_yaml('configs/example.yml')
app = create_app(config)
```

路由规则：

| YAML | 实际路径 | 行为 |
| --- | --- | --- |
| `add_sync: true` | `{prefix}{path}/sync` | `SyncServer` 阻塞等到模型返回 |
| `add_async: true` | `{prefix}{path}/async` | `AsyncServer` 丢进线程池，立刻返回队列状态 |
| 都未开 | `{prefix}{path}` | 直接把 `model` 当处理函数 |
| `app_func` | 原路径 | 自定义挂载（文档、GET、流式） |
| `apply: false` | — | 跳过 |

`model_configs` 会传给 `Model.from_configs`。若配置了 `base_model` 且该 key 已在前面初始化，会把那个实例传进去（Triton 热加载用）。

### SyncServer / AsyncServer / BaseServer

```python
from components.api import SyncServer, AsyncServer


def fake_model(data, **kwargs):
    return {'task_id': data['task_id'], 'data': data.get('kwargs')}


sync = SyncServer(fake_model)
print(sync({'task_id': 't1', 'kwargs': {'x': 1}}))

# async 立即返回，不包含最终业务结果
# AsyncServer(fake_model, n_pool=2, logger=some_logger)({'task_id': 't1'})
```

异步返回大致为 `{'task_id', 'data': {'state', 'wait_queue'}}`。最终结果靠落库 + 查询接口拿。

`BaseServer` 在普通模块生命周期外包了 `on_receive_start` / `on_respond_end`，一般不必直接继承。

### add_docs / add_ht / add_info_api / add_middleware / add_process_time_header / add_webui

在 YAML 的 `app_func` 或 `wrap_funcs` 里引用：

```yaml
/docs:
  title: Demo
  app_func: components.api.add_docs
```

- `add_docs`：Swagger UI，静态文件来自 `components/static`
- `add_ht`：`GET {path}/ht` 健康检查
- `add_info_api`：列出各模型 `module_info()`
- `add_middleware`：CORS 全开
- `add_process_time_header`：响应头 `X-Process-Time`
- `add_webui`：把 Gradio 挂到 FastAPI（`webui_app_func` 返回 gradio app）

### simple_get_router

无请求体的 GET。`func_configs.obj` 会传给模型。

```yaml
/ping:
  method: get
  app_func: components.api.simple_get_router
  model_instance: mypkg.models.ping.Model
  func_configs:
    obj:
      task_id: ''
      kwargs: {}
```

### stream_post_router

POST 后把模型的异步/同步生成器包成 `StreamingResponse`。

```yaml
/chat/stream:
  app_func: components.api.stream_post_router
  model_instance: mypkg.models.chat.AsyncModel
  request_template: mypkg.templates.chat.Request
```

模型应 `yield` 字符串（或 JSON 行）。

---

## 6. 客户端 `sdks`

密钥用环境变量或构造参数，不要写进代码和文档。

### openai.Openai

OpenAI 兼容 Chat Completions。`obj['post_kwargs']` 需含 `messages`。

```python
from components.sdks.openai import Openai

client = Openai(
    model='demo-model',
    client_kwargs=dict(api_key='***', base_url='http://127.0.0.1:8000/v1'),
)
ret = client({'post_kwargs': {'messages': [
    {'role': 'user', 'content': 'ping'},
]}})
print(ret['content'])
```

流式：`async for chunk in client.on_stream_process({'post_kwargs': {...}})`。  
超长输入可设 `max_input_length`；`finish_reason == content_filter` 会抛 `LLMBlockException`。

### openai.OpenaiMysqlCallbackModule

调用成功后把 messages、content、token 用量、耗时写入 `mysql_table`。过滤条件默认 `pid` / `sid`。

```python
from components.sdks.openai import OpenaiMysqlCallbackModule

llm = OpenaiMysqlCallbackModule(
    model='demo-model',
    mysql_table='llm_call',
    client_kwargs=dict(api_key='***', base_url='http://127.0.0.1:8000/v1'),
)
llm(
    {'post_kwargs': {'messages': [{'role': 'user', 'content': 'hi'}]}},
    pid=1,
    sid='step-1',
    task_id='t1',
)
```

### openai.Volcengine

对 `Openai` 的薄封装：`request(sys, user)` 拼 system/user 消息。默认 `disable_thinking=True`。连接类错误会重试。

```python
from components.sdks.openai import Volcengine

class Chat(Volcengine):
    model = 'demo-model'
    client_kwargs = dict(api_key='***', base_url='https://example.com/v1')

text = Chat().request('You are a helper.', 'Say hi.')
```

`return_content=False` 时返回完整 `obj`。`global_kwargs` 会传给内部 client（如 `pid` / `sid` / `task_id`）。

### openai.VolcengineMysqlModule

与上相同，但内部换成会落库的 client。必须设 `llm_cacher_mysql_table`。

```python
from components.sdks.openai import VolcengineMysqlModule
from workflows import skeletons


class LlmStep(VolcengineMysqlModule):
    model = 'demo-model'
    llm_cacher_mysql_table = 'llm_call'

    def on_process(self, obj, task_id=None, **kwargs):
        obj = dict(obj)
        obj['content'] = self.request(
            'You are a helper.',
            obj['text'],
            global_kwargs=dict(pid=obj.get('id'), task_id=task_id, sid='main'),
        )
        return obj
```

### openai.QwenVl

多模态：传入图片 URL。

```python
from components.sdks.openai import QwenVl

class VL(QwenVl):
    model = 'qwen-vl-max-latest'
    client_kwargs = dict(api_key='***', base_url='https://example.com/v1')

# VL().request('https://example.com/a.png')
```

### volcengine.Model / DoubaoSeed

火山方舟官方 SDK（`volcengine-python-sdk[ark]`）。`DoubaoSeed.request` 可拼文本、图片、音视频（本地文件会先 `files.create`，用完删除）。

```python
from components.sdks.volcengine import DoubaoSeed

class MM(DoubaoSeed):
    model = 'demo-model'

# MM().request(user='describe this', image_url='https://example.com/a.png')
```

### volcengine.MultiOpenaiAccountWrapper

`SwitchPipeline`：对多个 `api_key` 深拷贝同一客户端，按当前 running/queued 任务数选最空的账号。

```python
from components.sdks.volcengine import DoubaoSeed, MultiOpenaiAccountWrapper

class Pool(MultiOpenaiAccountWrapper):
    account_api_keys = ['key-a', 'key-b']

# wrapped = Pool(DoubaoSeed(model='demo-model'))
```

### bailian.Model

DashScope `Generation.call`。`llm_request(**post_kwargs)` 返回 content 字符串。

```python
from components.sdks.bailian import Model

m = Model(api_key='***', model='qwen-max')
# m.llm_request(messages=[{'role': 'user', 'content': 'hi'}])
```

### simple_client.BaseImageRequestClient

把 numpy 图转 base64，POST 到 `request_url`，校验 `code == 200` 后返回 `data`。子类只需提供 URL。

```python
from components.sdks.simple_client import BaseImageRequestClient


class EmbedClient(BaseImageRequestClient):
    @property
    def request_url(self):
        return 'http://127.0.0.1:8000/embed/sync'


# data = EmbedClient().request(images=['base64...'], task_id='t1')
```

### triton.TritonModule

Triton HTTP 客户端。子类实现 `request(obj, trt_client)`，按 `batch_size` 组 `async_infer`。`load` / `unload` 控制远端模型。

```python
from components.sdks.triton import TritonModule


class Infer(TritonModule):
    trt_url = '127.0.0.1:8000'
    trt_model_name = 'demo'
    batch_size = 4
    config_dir = 'configs'
    config_file = 'triton.yml'

    def request(self, obj, trt_client, **kwargs):
        async_req = trt_client.async_infer(
            obj['tensor'],
            model_name=self.trt_model_name,
        )
        outputs = trt_client.async_get(async_req)
        obj = dict(obj)
        obj['outputs'] = outputs
        return obj
```

`create_app` 里可用 `model_configs.base_model` 指向已创建的主模型，再挂加载/卸载接口：

```python
from components.sdks.triton import LoadTritonModule, UnloadTritonModule

class Load(LoadTritonModule):
    triton_module_name = 'Infer'

class Unload(UnloadTritonModule):
    triton_module_name = 'Infer'
```

YAML 示例：先声明带 `Infer` 子模块的主模型，后声明 `Load`，并设 `base_model` 为前者的路由 key。

### comfyui.Model

连本地 ComfyUI：提交 `prompt` 图，WebSocket 等到执行结束，再拉 history。`request_images` 会把输出图字节填进 `image_data`。

```python
from components.sdks.comfyui import Model

m = Model(host='127.0.0.1', port=8188)
# history = m.request(prompt={...})
# images = m.request_images(prompt={...})
```

### stable_diffusion_webui.Model

读 WebUI 的 `openapi.json`，可列出路径、生成 POST 示例、调用 `txt2img` / `img2img`（返回 numpy 图列表）。

```python
from components.sdks.stable_diffusion_webui import Model

m = Model(host='127.0.0.1', port=7860)
print(m.get_apis('sdapi')[:3])
# images = m.txt2img(prompt='a cat', steps=4)
```

---

## 7. 怎么往上堆

无库同步接口：

```python
from components.base import BaseModelWithoutDb
from components.template import BaseRequest, BaseSuccessResponse
from pydantic import BaseModel
from workflows import skeletons


class ReqData(BaseModel):
    text: str = ''


class Request(BaseRequest):
    kwargs: ReqData


class Response(BaseSuccessResponse):
    pass


class Work(skeletons.Module):
    def on_process(self, obj, **kwargs):
        obj = dict(obj)
        obj['text'] = obj['text'].upper()
        return obj


class Model(BaseModelWithoutDb):
    def __init__(self, cfgs={}, **kwargs):
        super().__init__(Work.from_configs(cfgs), **kwargs)
```

需要进度和失败可见时：任务类用 `BaseModelWithMysqlDb`，关键步骤用 `MysqlCallbackModule`，对外再配 `fetch` 查询节点（`MysqlDbModule.cacher.get_one`）。

选择建议：

| 需求 | 用 |
| --- | --- |
| HTTP 入参/出参 | `template.BaseRequest` / `BaseSuccessResponse` |
| 无状态任务 | `BaseModelWithoutDb` |
| 按 task_id 续跑、可查 | `BaseModelWithMysqlDb` + 子模块 `MysqlCallbackModule` |
| 结束后通知调用方 | `UrlCallbackModule` |
| YAML 自动出 `/sync` `/async` | `api.create_app` |
| OpenAI 兼容对话 | `sdks.openai.Volcengine` 或 `Openai` |
| 调用明细落库 | `VolcengineMysqlModule` / `OpenaiMysqlCallbackModule` |
| 图片 HTTP 推理 | `simple_client.BaseImageRequestClient` |
| Triton | `TritonModule` |
