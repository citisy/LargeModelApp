from . import openai


class Model(openai.Base):
    model: str
    ak: str
    sk: str
    base_url = 'https://ark.cn-beijing.volces.com/api/v3'
    client_kwargs = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from volcenginesdkarkruntime import Ark  # pip install --upgrade 'volcengine-python-sdk[ark]'

        self.client = Ark(
            base_url=self.base_url,
            ak=self.ak,
            sk=self.sk,
            **self.client_kwargs
        )

    def llm_request(self, **post_kwargs):
        post_result = self.client.chat.completions.create(
            model=self.model,
            **post_kwargs
        )
        content = post_result.choices[0].message.content
        return content


class ModelV2(openai.Model):
    """Base on openai"""
