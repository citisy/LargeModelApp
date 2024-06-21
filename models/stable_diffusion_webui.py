import re
import requests
import warnings
import numpy as np
from typing import List
from utils import configs, converter


class Model:
    """base on api of stable-diffusion-webui
    https://github.com/AUTOMATIC1111/stable-diffusion-webui"""

    def __init__(self, host='127.0.0.1', port=7860):
        self.host = host
        self.port = port

        url = f'http://{self.host}:{self.port}/openapi.json'
        r = requests.get(url)
        self.cache = r.json()

    def get_apis(self, keys: List[str] or str = None):
        """"
        >>> model = Model()
        >>> model.get_apis('sdapi')
        ['/sdapi/v1/txt2img', '/sdapi/v1/img2img', ...]
        """
        if keys:
            if isinstance(keys, str):
                keys = [keys]

            apis = []
            for k in self.cache['paths'].keys():
                for kk in keys:
                    if kk in k:
                        apis.append(k)
                        break
        else:
            apis = list(self.cache['paths'].keys())
        return apis

    def get_example_post_input(self, path):
        api_info = self.cache['paths'][path]
        if 'post' in api_info:
            schema = api_info['post']['requestBody']['content']['application/json']['schema']['$ref']
            p = self.cache
            for s in schema.split('/')[1:]:
                p = p[s]
            ret = configs.parse_pydantic_schema(p, {})
            ret = configs.parse_pydantic_dict(ret, return_default_value=True)
            return ret

        else:
            warnings.warn('can not find post method')
            return {}

    def get_example_post_output(self, path):
        pass

    def __call__(self, path, **post_kwargs):
        kwargs = self.get_example_post_input(path)
        kwargs.update(post_kwargs)
        url = f'http://{self.host}:{self.port}/{path}'
        url = re.sub(r'([^:])//+', r'\1/', url)
        r = requests.post(url, json=kwargs)
        if r.status_code == 200:
            return r.json()
        else:
            raise Exception(r.json())

    def txt2img(self, **post_kwargs) -> List[np.ndarray]:
        js = self('/sdapi/v1/txt2img', **post_kwargs)

        images = []
        for image in js['images']:
            image = converter.DataConvert.base64_to_image(image)
            images.append(image)

        return images

    def img2img(self, **post_kwargs) -> List[np.ndarray]:
        js = self('/sdapi/v1/img2img', **post_kwargs)

        images = []
        for image in js['images']:
            image = converter.DataConvert.base64_to_image(image)
            images.append(image)

        return images
