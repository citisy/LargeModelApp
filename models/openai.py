from workflows.skeletons import Module


class Base(Module):
    def request(self, *args, message_kwargs=dict(), post_kwargs=dict()):
        messages = self.make_message(*args, **message_kwargs)
        content = self.llm_request(messages=messages, **post_kwargs)
        return content

    def make_message(self, sys, user, *args, **kwargs):
        return [
            {
                "role": "system",
                "content": sys,
            },
            {
                "role": "user",
                "content": user,
            },
        ]

    def llm_request(self, **post_kwargs):
        raise NotImplementedError


class Model(Base):
    model: str = 'gpt-4'
    client_kwargs: dict = dict(
        api_key=...,
        base_url=...
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from openai import OpenAI  # pip install openai
        self.client = OpenAI(**self.client_kwargs)

    def llm_request(self, **post_kwargs):
        post_result = self.client.chat.completions.create(
            model=self.model,
            **post_kwargs
        )
        content = post_result.choices[0].message.content
        return content
