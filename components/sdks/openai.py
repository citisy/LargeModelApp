import asyncio
import os

import openai

from workflows import exceptions, skeletons, callbacks
from .. import _callbacks, base


class Base(skeletons.Module):
    def on_process_end(self, obj, **kwargs):
        post_result = obj['post_result']
        content = post_result.choices[0].message.content
        obj.update(content=content)

        return obj


class Openai(Base):
    model: str
    client_kwargs: dict = {}

    max_input_length = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from openai import AsyncOpenAI  # pip install openai
        self.client = AsyncOpenAI(**self.client_kwargs)

    def on_process(self, obj, **kwargs):
        # important post_kwargs: [messages]
        post_kwargs = obj['post_kwargs']

        if self.max_input_length and 'messages' in post_kwargs and len(str(post_kwargs['messages'])) > self.max_input_length:
            raise exceptions.LLMInputOutOfLengthException(len(str(post_kwargs['messages'])), self.max_input_length)

        post_result = asyncio.run(self.client.chat.completions.create(
            model=self.model,
            **post_kwargs
        ))

        if post_result.choices[0].finish_reason == 'content_filter':
            raise exceptions.LLMBlockException()

        obj.update(post_result=post_result)
        return obj

    async def on_stream_process(self, obj, **kwargs):
        # important post_kwargs: [messages]
        post_kwargs = obj['post_kwargs']

        async for chunk in await self.client.chat.completions.create(
                model=self.model,
                stream=True,
                **post_kwargs
        ):
            content = chunk.choices[0].delta.content
            if content:
                yield content


class OpenaiMysqlCallbackModule(Openai, base.MysqlCallbackModule):
    add_url_callback = False

    mysql_cacher_keys: list = ['model', 'messages', 'content', 'reasoning_content', 'total_tokens', 'prompt_tokens', 'completion_tokens', 'reasoning_tokens']
    mysql_filter_mapping = {'pid': 'pid', 'sid': 'sid'}  # (mysql_key, obj_key)

    global_cacher_keys = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.callback_wrapper.register_success_callback(
            callbacks.TimeLoggerCallback(
                fmt='[{pid}-{sid}] request ' + self.model + ' takes {time:.2f} s'
            )
        )

        self.callback_wrapper.register_success_callback(
            _callbacks.TimeDbCacheCallback(
                mysql_table=self.mysql_table,
                db_filter_mapping=self.mysql_filter_mapping,
                cache_key='duration'
            )
        )
        self.ignore_errors = False

    def gen_kwargs(self, obj, **kwargs):
        kwargs = super().gen_kwargs(obj, **kwargs)
        kwargs.setdefault('sid', None)
        return kwargs

    def on_process_start(self, obj, pid=None, sid=None, **kwargs):
        obj.update(
            pid=pid,
            sid=sid,
            model=self.model,
        )
        return obj

    def on_process_end(self, obj, **kwargs):
        post_kwargs = obj['post_kwargs']
        post_result = obj['post_result']

        messages = post_kwargs['messages']

        message = post_result.choices[0].message
        content = message.content
        reasoning_content = message.model_extra.get('reasoning_content', '')

        usage = post_result.usage
        completion_tokens = usage.completion_tokens
        prompt_tokens = usage.prompt_tokens
        total_tokens = usage.total_tokens
        reasoning_tokens = usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details else 0

        obj.update(
            messages=messages,
            content=content,
            reasoning_content=reasoning_content,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
        )

        return obj


class Volcengine(skeletons.RetryModule):
    model: str

    client_kwargs = dict(
        api_key=os.getenv('VOL_API_KEY', ""),
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    llm_client_kwargs = {}
    disable_thinking = True

    err_type = (ConnectionError, openai.APIConnectionError, exceptions.LLMParseException)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_client = Openai(
            model=self.model,
            client_kwargs=self.client_kwargs,
            **self.llm_client_kwargs
        )

    def request(
            self, sys=None, user=None, messages=None,
            return_content=True, global_kwargs={}, **post_kwargs
    ):
        messages = messages or [
            {
                "role": "system",
                "content": sys,
            },
            {
                "role": "user",
                "content": user,
            },
        ]

        if self.disable_thinking:
            post_kwargs.setdefault(
                'extra_body',
                dict(
                    thinking={
                        "type": "disabled"
                    },
                    chat_template_kwargs={"enable_thinking": False},
                    enable_thinking=False
                )
            )

        ret = self.llm_client(dict(
            post_kwargs=dict(
                messages=messages,
                **post_kwargs
            ),
        ), **global_kwargs)
        if return_content:
            return ret['content']
        else:
            return ret

    def on_stream_request(self, sys, user, **post_kwargs):
        messages = [
            {
                "role": "system",
                "content": sys,
            },
            {
                "role": "user",
                "content": user,
            },
        ]

        return self.llm_client.on_stream_process(dict(
            post_kwargs=dict(
                messages=messages,
                **post_kwargs
            )
        ))


class VolcengineMysqlModule(Volcengine):
    """
    Usage:
        class LlmRequest(VolcengineDbCallbackModule):
            def on_process(self, obj, task_id=None, **kwargs):
                ...
                llm_result = self.request(
                    sys,
                    user,
                    global_kwargs=dict(
                        pid=obj['id'],
                        task_id=task_id,
                        sid="xxx"
                    )
                )
                ...
                return obj

    """
    llm_cacher_mysql_table: str
    max_input_length: int = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_client = OpenaiMysqlCallbackModule(
            model=self.model,
            client_kwargs=self.client_kwargs,
            mysql_table=self.llm_cacher_mysql_table,
            max_input_length=self.max_input_length
        )


class QwenVl(skeletons.RetryModule):
    model = "qwen-vl-max-latest"

    client_kwargs = dict()
    llm_client_kwargs = {}
    err_type = (ConnectionError, openai.APIConnectionError, exceptions.LLMParseException)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_client = Openai(
            model=self.model,
            client_kwargs=self.client_kwargs,
            **self.llm_client_kwargs
        )

    def request(self, img_url, **post_kwargs):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url
                        },
                    },
                    {"type": "text", "text": "这张图片描述了些什么？"},
                ],
            }
        ]

        ret = self.llm_client(dict(
            post_kwargs=dict(
                messages=messages,
                **post_kwargs
            )
        ))
        return ret['content']
