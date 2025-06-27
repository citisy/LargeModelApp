from http import HTTPStatus

from dashscope import Generation  # pip install dashscope

from . import openai


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


class ModelV2(openai.Model):
    """Base on openai"""


class QwenVl(openai.Model):
    def make_message(self, image_obj, user, *args, **kwargs):
        """
        image_obj can be an url like "http://xxx"
        a file path like "file://xxx"
        a base64 str like "data:image/png;base64,xxx"
        """
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_obj
                        },
                    },
                    {"type": "text", "text": user},
                ],
            }
        ]
