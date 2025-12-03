from http import HTTPStatus

from dashscope import Generation  # pip install dashscope


class Model(openai.Base):
    model: str = 'qwen-max'
    api_key: str

    def llm_request(self, **post_kwargs):
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
        content = post_result.choices[0].message.content
        return content



