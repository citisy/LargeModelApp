import asyncio
import copy
import os

import numpy as np

from components.sdks import openai
from utils import converter, os_lib
from workflows import skeletons


class Model(openai.Base):
    model: str
    ak: str
    sk: str
    base_url = 'https://ark.cn-beijing.volces.com/api/v3'
    client_kwargs = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from volcenginesdkarkruntime import Ark  # pip install --upgrade 'volcengine-python-sdk[ark]'

        self.llm_client = Ark(
            base_url=self.base_url,
            ak=self.ak,
            sk=self.sk,
            **self.client_kwargs
        )

    def request(self, **post_kwargs):
        post_result = self.llm_client.chat.completions.create(
            model=self.model,
            **post_kwargs
        )
        content = post_result.choices[0].message.content
        return content


class DoubaoSeed(skeletons.RetryModule):
    model: str

    client_kwargs = dict(
        api_key=os.getenv('VOL_API_KEY', ''),
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    disable_thinking = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from volcenginesdkarkruntime import Ark

        self.llm_client = Ark(
            **self.client_kwargs
        )

    def request(
            self,
            sys=None, user=None,
            image_array: np.ndarray = None, image_url: str = None,
            image_arrays: list = None, image_urls: list = None,
            video_path: str = None, video_url: str = None,
            video_paths: str = None, video_urls: str = None,
            audio_path: str = None, audio_url: str = None,
            audio_paths: str = None, audio_urls: str = None,
            content=None,
            upload_kwargs={}, return_content=True, **post_kwargs
    ):
        content = content or []
        file_id = None

        if sys is not None:
            content.append(
                {
                    "type": "input_text",
                    "text": sys
                }
            )

        if user is not None:
            content.append(
                {
                    "type": "input_text",
                    "text": user
                }
            )

        if image_array is not None:
            image = converter.DataConvert.image_to_base64(image_array)
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image}"
            })

        if image_url is not None:
            content.append({
                "type": "input_image",
                "image_url": image_url
            })

        if image_arrays is not None:
            for image_array in image_arrays:
                image = converter.DataConvert.image_to_base64(image_array)
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image}"
                })

        if image_urls is not None:
            for image_url in image_urls:
                content.append({
                    "type": "input_image",
                    "image_url": image_url
                })

        if video_path is not None:
            file_id = self.upload_file(video_path, **upload_kwargs)
            content.append({
                "type": "input_video",
                "file_id": file_id
            })

        if video_url is not None:
            content.append({
                "type": "input_video",
                "video_url": video_url
            })

        if video_paths is not None:
            for video_path in video_paths:
                file_id = self.upload_file(video_path, **upload_kwargs)
                content.append({
                    "type": "input_video",
                    "file_id": file_id
                })

        if video_urls is not None:
            for video_url in video_urls:
                content.append({
                    "type": "input_video",
                    "video_url": video_url
                })

        if audio_path is not None:
            file_id = self.upload_file(audio_path, **upload_kwargs)
            content.append({
                "type": "input_audio",
                "file_id": file_id
            })

        if audio_url is not None:
            content.append({
                "type": "input_audio",
                "audio_url": audio_url
            })

        if audio_paths is not None:
            for audio_path in audio_paths:
                file_id = self.upload_file(audio_path, **upload_kwargs)
                content.append({
                    "type": "input_audio",
                    "file_id": file_id
                })

        if audio_urls is not None:
            for audio_url in audio_urls:
                content.append({
                    "type": "input_audio",
                    "audio_url": audio_url
                })

        if self.disable_thinking:
            # 统一默认关闭思考模式
            post_kwargs.setdefault(
                'extra_body',
                {
                    "reasoning": {
                        'effort': 'minimal'
                    }
                }
            )

        response = self.llm_client.responses.create(
            model=self.model,
            input=[
                {"role": "user", "content": content},
            ],
            **post_kwargs
        )

        if file_id is not None:
            self.llm_client.files.delete(
                file_id=file_id
            )
        if return_content:
            return response.output[-1].content[0].text
        else:
            return response

    def upload_file(self, path, upload_kwargs={}):
        file = self.llm_client.files.create(
            file=os_lib.loader.load_bytes(path),
            purpose="user_data",
            **upload_kwargs
        )

        self.llm_client.files.wait_for_processing(file.id)

        return file.id


class MultiOpenaiAccountWrapper(skeletons.SwitchPipeline):
    """
    Usages:
        module = Volcengine(...)
        module = MultiOpenaiAccountWrapper(module)
    """
    account_api_keys: list

    def __init__(self, base_module: DoubaoSeed, **kwargs):
        modules = []
        for api_key in self.account_api_keys:
            module = copy.deepcopy(base_module)
            module.llm_client.api_key = api_key
            modules.append(module)

        super().__init__(modules, **kwargs)

    def switch(self, obj, **kwargs) -> str | int:
        from volcenginesdkarkruntime import AsyncArk

        async def choose_account():
            clients = [AsyncArk(api_key=api_key) for api_key in self.account_api_keys]
            try:
                running_responses = await asyncio.gather(*[
                    client.content_generation.tasks.list(
                        page_size=15,
                        status="running",
                    )
                    for client in clients
                ])
                running_counts = [len(response.items) for response in running_responses]

                if min(running_counts) >= 10:
                    queued_responses = await asyncio.gather(*[
                        client.content_generation.tasks.list(
                            page_size=50,
                            status="queued",
                        )
                        for client in clients
                    ])
                    queued_counts = [len(response.items) for response in queued_responses]
                    return queued_counts.index(min(queued_counts))

                return running_counts.index(min(running_counts))
            finally:
                await asyncio.gather(*[client.close() for client in clients], return_exceptions=True)

        account_type = asyncio.run(choose_account())
        obj.update(account_type=account_type)
        return account_type
