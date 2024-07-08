# pip install dashscope
from dashscope import Generation
from http import HTTPStatus
from . import openai


class Model(openai.Base):
    api_key: str
    llm_type: str = 'qwen-max'

    def on_process_start(self, obj, **kwargs):
        # important post_kwargs: [prompt, messages, history]
        post_kwargs = obj['post_kwargs']

        response = Generation.call(
            self.llm_type,
            api_key=self.api_key,
            result_format='message',  # 设置输出为'message'格式
            **post_kwargs
        )

        assert response.status_code == HTTPStatus.OK, 'Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        )

        post_result = response.output
        obj.update(post_result=post_result)
        return obj
