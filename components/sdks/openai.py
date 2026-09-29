import asyncio
import os

import openai

from workflows import exceptions, skeletons, callbacks
from .. import _callbacks, base
import time


class ChatClient(skeletons.RetryModule):
    model: str
    client_kwargs: dict = {}

    max_input_length = None
    disable_thinking = True

    err_type = (ConnectionError, openai.APIConnectionError, exceptions.LLMParseException)

    @property
    def client(self):
        from openai import AsyncOpenAI  # pip install openai
        return AsyncOpenAI(**self.client_kwargs)

    def make_post_kwargs(self, sys=None, user=None, messages=None, post_kwargs=dict()):
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
        if self.max_input_length and len(str(messages)) > self.max_input_length:
            raise exceptions.LLMInputOutOfLengthException(len(str(messages)), self.max_input_length)

        post_kwargs.update(
            model=self.model,
            messages=messages,
        )

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

        return post_kwargs

    async def run(self, **post_kwargs):
        # solve: RuntimeError: Event loop is closed
        async with self.client as client:
            post_result = await client.chat.completions.create(**post_kwargs)

        if post_result.choices[0].finish_reason == 'content_filter':
            raise exceptions.LLMBlockException()

        return post_result

    def request(
            self, sys=None, user=None, messages=None,
            return_content=True, **post_kwargs
    ):
        post_kwargs = self.make_post_kwargs(sys=sys, user=user, messages=messages, post_kwargs=post_kwargs)
        post_result = asyncio.run(self.run(**post_kwargs))

        if return_content:
            return post_result.choices[0].message.content
        else:
            return post_result

    async def on_stream_request(self, sys=None, user=None, messages=None, **post_kwargs):
        post_kwargs = self.make_post_kwargs(sys=sys, user=user, messages=messages, post_kwargs=post_kwargs)

        async for chunk in await self.client.chat.completions.create(
                stream=True,
                **post_kwargs
        ):
            content = chunk.choices[0].delta.content
            if content:
                yield content


class ResponseClient(skeletons.RetryModule):
    model: str
    client_kwargs: dict = {}

    max_input_length = None
    disable_thinking = True

    err_type = (ConnectionError, openai.APIConnectionError, exceptions.LLMParseException)

    @property
    def client(self):
        from openai import AsyncOpenAI  # pip install openai
        return AsyncOpenAI(**self.client_kwargs)

    def make_post_kwargs(self, sys=None, user=None, messages=None, post_kwargs=dict()):
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
        if self.max_input_length and len(str(messages)) > self.max_input_length:
            raise exceptions.LLMInputOutOfLengthException(len(str(messages)), self.max_input_length)

        post_kwargs.update(
            model=self.model,
            input=messages,
        )

        return post_kwargs

    async def run(self, **post_kwargs):
        # solve: RuntimeError: Event loop is closed
        async with self.client as client:
            return await client.responses.create(**post_kwargs)

    def request(
            self, sys=None, user=None, messages=None,
            **post_kwargs
    ):
        post_kwargs = self.make_post_kwargs(sys=sys, user=user, messages=messages, post_kwargs=post_kwargs)
        return asyncio.run(self.run(**post_kwargs))

    async def on_stream_request(self, sys=None, user=None, messages=None, **post_kwargs):
        post_kwargs = self.make_post_kwargs(sys=sys, user=user, messages=messages, post_kwargs=post_kwargs)

        async for chunk in await self.client.responses.create(
                stream=True,
                **post_kwargs
        ):
            content = chunk.choices[0].delta.content
            if content:
                yield content


class VlClient(skeletons.RetryModule):
    model: str
    client_kwargs: dict = {}

    err_type = (ConnectionError, openai.APIConnectionError, exceptions.LLMParseException)

    @property
    def client(self):
        from openai import AsyncOpenAI  # pip install openai
        return AsyncOpenAI(**self.client_kwargs)

    def make_post_kwargs(self, img_url=None, messages=None, post_kwargs=dict()):
        messages = messages or [
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

        post_kwargs.update(
            model=self.model,
            messages=messages,
        )

        return post_kwargs

    async def run(self, **post_kwargs):
        # solve: RuntimeError: Event loop is closed
        async with self.client as client:
            post_result = await client.chat.completions.create(**post_kwargs)

        if post_result.choices[0].finish_reason == 'content_filter':
            raise exceptions.LLMBlockException()

        return post_result

    def request(self, img_url=None, messages=None, return_content=True, **post_kwargs):
        post_kwargs = self.make_post_kwargs(img_url=img_url, messages=messages, post_kwargs=post_kwargs)
        post_result = asyncio.run(self.run(**post_kwargs))

        if return_content:
            return post_result.choices[0].message.content
        else:
            return post_result


class ChatClientMysqlCallbackModule(ChatClient):
    mysql_callback_kwargs: dict

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mysql_callback_kwargs = dict(
            mysql_cacher_keys=['model', 'messages', 'content', 'reasoning_content', 'total_tokens', 'prompt_tokens', 'completion_tokens', 'reasoning_tokens'],
            mysql_filter_mapping={'pid': 'pid', 'sid': 'sid'},  # (mysql_key, obj_key)
            global_cacher_keys=[],
        )
        mysql_callback_kwargs.update(self.mysql_callback_kwargs)
        self.mysql_callback_module = base.MysqlCallbackModule(**mysql_callback_kwargs)
        self.ignore_errors = False

    def request(
            self, *args,
            pid=None, sid=None, return_content=True,
            **post_kwargs
    ):
        t1 = time.time()
        post_result = super().request(*args, return_content=False, **post_kwargs)
        t2 = time.time()
        message = post_result.choices[0].message
        content = message.content
        reasoning_content = message.model_extra.get('reasoning_content', '')

        usage = post_result.usage
        completion_tokens = usage.completion_tokens
        prompt_tokens = usage.prompt_tokens
        total_tokens = usage.total_tokens
        reasoning_tokens = usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details else 0

        self.mysql_callback_module(dict(
            pid=pid,
            sid=sid,
            model=self.model,
            messages=post_kwargs['messages'],
            content=content,
            reasoning_content=reasoning_content,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            duration=t2-t1
        ))

        if return_content:
            return post_result.choices[0].message.content
        else:
            return post_result


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


class Volcengine(ChatClient):
    client_kwargs = dict(
        api_key=os.getenv('VOL_API_KEY', ""),
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )


class VolcengineMysqlModule(ChatClientMysqlCallbackModule):
    """
    Usage:
        class LlmRequest(VolcengineMysqlModule):
            def on_process(self, obj, task_id=None, **kwargs):
                ...
                llm_result = self.request(
                    sys,
                    user,
                    pid=task_id,   # or use obj['id']
                    sid="xxx"
                )
                ...
                return obj

    """
    client_kwargs = dict(
        api_key=os.getenv('VOL_API_KEY', ""),
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
