import re
import warnings
from typing import List

import numpy as np
import requests

from utils import configs, converter
from workflows.skeletons import Module


class Model(Module):
    """base on apis of stable-diffusion-webui
    https://github.com/AUTOMATIC1111/stable-diffusion-webui"""
    host = '127.0.0.1'
    port = 7860

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        url = f'http://{self.host}:{self.port}/openapi.json'
        r = requests.get(url)
        self.cache = r.json()

    def make_tree_paths(self):
        tree_paths = {}
        for k, v in self.cache['paths'].items():
            p = tree_paths
            a = k.split('/')
            for aa in a[:-1]:
                p = p.setdefault(aa, {})

            p[a[-1]] = v

        return tree_paths

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
            schema = api_info['post']['requestBody']['content']['application/json']['schema']
            if '$ref' in schema:
                ref = schema['$ref']
                p = self.cache
                for s in ref.split('/')[1:]:
                    p = p[s]
                ret = configs.PydanticParse.parse_schema(p, {})
                ret = configs.PydanticParse.parse_ret_dict(ret, return_default_value=True)
                return ret
            else:
                return {}

        else:
            warnings.warn('can not find post method')
            return {}

    def request(self, path, **post_kwargs) -> dict:
        kwargs = self.get_example_post_input(path)
        kwargs.update(post_kwargs)
        url = f'http://{self.host}:{self.port}/{path}'
        url = re.sub(r'([^:])//+', r'\1/', url)
        r = requests.post(url, json=kwargs)
        assert r.status_code == 200, r.json()

        ret = r.json()

        return ret

    def txt2img(self, **post_kwargs) -> List[np.ndarray]:
        obj = self.request(
            path='/sdapi/v1/txt2img',
            **post_kwargs
        )

        ret = obj['ret']

        images = []
        for image in ret['images']:
            image = converter.DataConvert.base64_to_image(image)
            images.append(image)

        return images

    def img2img(self, **post_kwargs) -> List[np.ndarray]:
        obj = self.request(
            path='/sdapi/v1/img2img',
            **post_kwargs
        )

        ret = obj['ret']

        images = []
        for image in ret['images']:
            image = converter.DataConvert.base64_to_image(image)
            images.append(image)

        return images
