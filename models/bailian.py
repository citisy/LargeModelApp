from dashscope import Generation    # pip install dashscope
from http import HTTPStatus
from . import openai


class Model(openai.Base):
    model: str = 'qwen-max'
    api_key: str

    def on_process(self, obj, **kwargs):
        # important post_kwargs: [prompt, messages, history]
        post_kwargs = obj['post_kwargs']

        response = Generation.call(
            self.model,
            api_key=self.api_key,
            result_format='message',
            **post_kwargs
        )

        assert response.status_code == HTTPStatus.OK, 'Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        )

        post_result = response.output
        obj.update(post_result=post_result)
        return obj
