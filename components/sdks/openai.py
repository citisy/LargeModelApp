import asyncio
import os

import openai

from workflows import exceptions, skeletons, callbacks
from .. import _callbacks, base
import time
from utils import converter, os_lib
from pathlib import Path


class ChatClient(skeletons.RetryModule):
    model: str
    client_kwargs: dict = {}

    max_input_length = None
    disable_thinking = False

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

    def collect(self, post_kwargs, post_result, t1, t2):
        message = post_result.choices[0].message
        content = message.content
        reasoning_content = message.model_extra.get('reasoning_content', '')

        usage = post_result.usage
        completion_tokens = usage.completion_tokens
        prompt_tokens = usage.prompt_tokens
        total_tokens = usage.total_tokens
        reasoning_tokens = usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details else 0

        return dict(
            model=self.model,
            messages=post_kwargs['messages'],
            content=content,
            reasoning_content=reasoning_content,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            duration=t2 - t1
        )

    def request(
            self,
            sys=None, user=None, messages=None,
            return_content=True, return_collection=False,
            **post_kwargs
    ):
        t1 = time.time()
        post_kwargs = self.make_post_kwargs(sys=sys, user=user, messages=messages, post_kwargs=post_kwargs)
        post_result = asyncio.run(self.run(**post_kwargs))
        t2 = time.time()

        if return_content:
            results = post_result.choices[0].message.content
        else:
            results = post_result

        if return_collection:
            collection = self.collect(post_kwargs, post_result, t1, t2)
            results = (results, collection)

        return results

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
    disable_thinking = False

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

    def collect(self, post_kwargs, post_result, t1, t2):
        content = post_result.output_text or ''
        usage = post_result.usage
        prompt_tokens = getattr(usage, 'input_tokens', 0) if usage else 0
        completion_tokens = getattr(usage, 'output_tokens', 0) if usage else 0
        total_tokens = getattr(usage, 'total_tokens', 0) if usage else 0

        return dict(
            model=self.model,
            input=post_kwargs['input'],
            content=content,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration=t2 - t1
        )

    def request(
            self, sys=None, user=None, messages=None,
            return_content=True, return_collection=False,
            **post_kwargs
    ):
        t1 = time.time()
        post_kwargs = self.make_post_kwargs(sys=sys, user=user, messages=messages, post_kwargs=post_kwargs)
        post_result = asyncio.run(self.run(**post_kwargs))
        t2 = time.time()

        if return_content:
            results = post_result.output_text
        else:
            results = post_result

        if return_collection:
            collection = self.collect(post_kwargs, post_result, t1, t2)
            results = (results, collection)

        return results

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

    def make_post_kwargs(
            self,
            prompt=None,
            image_url=None, image_urls=None,
            image_path=None, image_paths=None,
            messages=None,
            post_kwargs=dict()
    ):
        if messages is None:
            prompt = prompt or "What does this picture describe?"

            content = [
                {"type": "input_text", "text": prompt},
            ]

            if image_url is not None:
                content.append(
                    {"type": "input_image", "image_url": {"url": image_url}}
                )

            if image_urls is not None:
                for image_url in image_urls:
                    content.append(
                        {"type": "input_image", "image_url": {"url": image_url}}
                    )

            if image_path is not None:
                base64_image = converter.DataConvert.bytes_to_base64(os_lib.loader.load_bytes(image_path))
                mime_type = Path(image_path).suffix[1:]
                content.append(
                    {"type": "input_image", "image_url": f"data:image/{mime_type};base64,{base64_image}"}
                )

            if image_paths is not None:
                for image_path in image_paths:
                    base64_image = converter.DataConvert.bytes_to_base64(os_lib.loader.load_bytes(image_path))
                    mime_type = Path(image_path).suffix[1:]
                    content.append(
                        {"type": "input_image", "image_url": f"data:image/{mime_type};base64,{base64_image}"}
                    )

            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

        post_kwargs.update(
            model=self.model,
            input=messages,
        )

        return post_kwargs

    async def run(self, **post_kwargs):
        # solve: RuntimeError: Event loop is closed
        async with self.client as client:
            post_result = await client.responses.create(**post_kwargs)

        return post_result

    def collect(self, post_kwargs, post_result, t1, t2):
        content = post_result.output_text or ''
        usage = post_result.usage
        prompt_tokens = getattr(usage, 'input_tokens', 0) if usage else 0
        completion_tokens = getattr(usage, 'output_tokens', 0) if usage else 0
        total_tokens = getattr(usage, 'total_tokens', 0) if usage else 0

        return dict(
            model=self.model,
            input=post_kwargs['input'],
            content=content,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration=t2 - t1
        )

    def request(
            self,
            prompt=None,
            image_url=None, image_urls=None,
            image_path=None, image_paths=None,
            messages=None,
            return_content=True, return_collection=False,
            **post_kwargs
    ):
        t1 = time.time()
        post_kwargs = self.make_post_kwargs(
            prompt=prompt,
            image_url=image_url, image_urls=image_urls,
            image_path=image_path, image_paths=image_paths,
            messages=messages,
            post_kwargs=post_kwargs
        )
        post_result = asyncio.run(self.run(**post_kwargs))
        t2 = time.time()

        if return_content:
            results = post_result.output_text
        else:
            results = post_result

        if return_collection:
            collection = self.collect(post_kwargs, post_result, t1, t2)
            results = (results, collection)
        return results


class Text2ImageClient(skeletons.RetryModule):
    model: str
    client_kwargs: dict = {}

    max_input_length = None

    err_type = (ConnectionError, openai.APIConnectionError, exceptions.LLMParseException)

    @property
    def client(self):
        from openai import OpenAI  # pip install openai
        return OpenAI(**self.client_kwargs)

    def make_post_kwargs(self, prompt=None, post_kwargs=dict()):
        post_kwargs.update(
            model=self.model,
            prompt=prompt,
        )

        return post_kwargs

    def run(self, **post_kwargs):
        return self.client.images.generate(**post_kwargs)

    def collect(self, post_kwargs, post_result, t1, t2):
        usage = getattr(post_result, 'usage', None)
        prompt_tokens = getattr(usage, 'input_tokens', 0) if usage else 0
        completion_tokens = getattr(usage, 'output_tokens', 0) if usage else 0
        total_tokens = getattr(usage, 'total_tokens', 0) if usage else 0

        first = post_result.data[0] if getattr(post_result, 'data', None) else None
        content = (getattr(first, 'url', '') or '') if first else ''
        revised_prompt = getattr(first, 'revised_prompt', '') if first else ''

        return dict(
            model=self.model,
            prompt=post_kwargs['prompt'],
            content=content,
            revised_prompt=revised_prompt,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration=t2 - t1
        )

    def request(
            self, prompt=None,
            return_collection=False,
            **post_kwargs
    ):
        t1 = time.time()
        post_kwargs = self.make_post_kwargs(prompt=prompt, post_kwargs=post_kwargs)
        post_result = self.run(**post_kwargs)
        t2 = time.time()

        results = post_result

        if return_collection:
            collection = self.collect(post_kwargs, post_result, t1, t2)
            results = (results, collection)

        return results



class Image2ImageClient(skeletons.RetryModule):
    model: str
    client_kwargs: dict = {}

    max_input_length = None

    err_type = (ConnectionError, openai.APIConnectionError, exceptions.LLMParseException)

    @property
    def client(self):
        from openai import OpenAI  # pip install openai
        return OpenAI(**self.client_kwargs)

    def make_post_kwargs(self, prompt=None, image_path=None, image_paths=None, post_kwargs=dict()):
        _image_path = []

        if image_path is not None:
            _image_path.append(image_path)
        if image_paths is not None:
            _image_path.extend(image_paths)

        images = []
        for source in _image_path:
            path = Path(source)
            mime = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                    '.png': 'image/png', '.webp': 'image/webp'}.get(path.suffix.lower())
            if mime is None:
                raise ValueError(f'Unsupported image format: {path.suffix}')
            # Explicit MIME prevents .webp being sent as application/octet-stream.
            images.append((path.name, path.read_bytes(), mime))

        post_kwargs.update(
            model=self.model,
            prompt=prompt,
            image=images,
        )

        return post_kwargs

    def run(self, **post_kwargs):
        return self.client.images.edit(**post_kwargs)

    def collect(self, post_kwargs, post_result, t1, t2):
        usage = getattr(post_result, 'usage', None)
        prompt_tokens = getattr(usage, 'input_tokens', 0) if usage else 0
        completion_tokens = getattr(usage, 'output_tokens', 0) if usage else 0
        total_tokens = getattr(usage, 'total_tokens', 0) if usage else 0

        first = post_result.data[0] if getattr(post_result, 'data', None) else None
        content = (getattr(first, 'url', '') or '') if first else ''
        revised_prompt = getattr(first, 'revised_prompt', '') if first else ''

        return dict(
            model=self.model,
            prompt=post_kwargs['prompt'],
            content=content,
            revised_prompt=revised_prompt,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration=t2 - t1
        )

    def request(
            self, prompt=None, image_path=None, image_paths=None,
            return_collection=False,
            **post_kwargs
    ):
        t1 = time.time()
        post_kwargs = self.make_post_kwargs(prompt=prompt, image_path=image_path, image_paths=image_paths, post_kwargs=post_kwargs)
        post_result = self.run(**post_kwargs)
        t2 = time.time()

        results = post_result

        if return_collection:
            collection = self.collect(post_kwargs, post_result, t1, t2)
            results = (results, collection)

        return results


class Volcengine(ChatClient):
    client_kwargs = dict(
        api_key=os.getenv('VOL_API_KEY', ""),
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
