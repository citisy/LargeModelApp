import asyncio

from workflows import exceptions, skeletons


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


class Volcengine(skeletons.Module):
    model: str

    client_kwargs = dict(
        api_key="",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_client = Openai(
            model=self.model,
            client_kwargs=self.client_kwargs
        )

    def request(self, sys, user, return_content=True, global_kwargs={}, **post_kwargs):
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


class QwenVl(skeletons.Module):
    model: str

    client_kwargs = dict(
        api_key='',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_client = Openai(
            model=self.model,
            client_kwargs=self.client_kwargs
        )

    def request(self, img_url, text="这张图片描述了些什么？", **post_kwargs):
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
                    {"type": "text", "text": text},
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

