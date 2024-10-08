from workflows.skeletons import Module


class Base(Module):
    def on_process_end(self, obj, **kwargs):
        post_result = obj['post_result']
        content = post_result.choices[0].message.content
        obj.update(content=content)

        return obj


class Model(Base):
    model: str = 'gpt-4'
    client_kwargs: dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from openai import OpenAI  # pip install openai
        self.client = OpenAI(**self.client_kwargs)

    def on_process(self, obj, **kwargs):
        # important post_kwargs: [messages]
        post_kwargs = obj['post_kwargs']

        post_result = self.client.chat.completions.create(
            model=self.model,
            **post_kwargs
        )

        obj.update(post_result=post_result)
        return obj
