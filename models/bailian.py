from dashscope import Generation
from http import HTTPStatus
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class LLM(LLM):
    api_key : str
    llm_type : str = 'qwen-max'

    def _llm_type(self) -> str:
        return self.llm_type

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        response = Generation.call(
            self.llm_type,
            prompt,
            api_key=self.api_key,
            result_format='message',  # 设置输出为'message'格式
        )

        if response.status_code == HTTPStatus.OK:
            text = response.output.choices[0]['message']['content']

        else:
            text = 'Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            )

        return text
