from typing import List

import numpy as np
import requests

from utils import converter
from workflows import skeletons


class BaseImageRequestClient(skeletons.RetryModule):
    err_type = requests.HTTPError

    @property
    def request_url(self):
        raise NotImplementedError

    def request(
            self,
            images: List[np.ndarray] | List[str] = None,
            task_id=None,
            **req_kwargs
    ):
        req_data = dict()

        if task_id is not None:
            req_data['task_id'] = task_id

        if images is not None:
            if isinstance(images[0], np.ndarray):
                images = [converter.DataConvert.image_to_base64(img) for img in images]
            req_kwargs['images'] = images

        req_data['kwargs'] = req_kwargs
        r = requests.post(self.request_url, json=req_data)
        r.raise_for_status()
        ret = r.json()

        assert ret['code'] == 200, ret['msg']
        return ret["data"]
