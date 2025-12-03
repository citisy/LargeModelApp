from tritonclient.utils import InferenceServerException

from utils import triton_utils
from workflows import skeletons


class TritonModule(skeletons.Module):
    trt_url: str
    trt_model_name: str

    model_versions: dict
    model_configs: dict

    batch_size = 4

    config_dir: str
    config_file: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config_path = f'{self.config_dir}/{self.config_file}' if hasattr(self, 'config_file') else ''

    @property
    def trt_client(self):
        _trt_client = triton_utils.HttpClient(url=self.trt_url, config_path=self.config_path)
        if hasattr(self, 'model_versions'):
            _trt_client.model_versions = self.model_versions
        if hasattr(self, 'model_configs'):
            _trt_client.model_configs = self.model_configs
        return _trt_client

    def load(self, **kwargs):
        try:
            self.trt_client.load(self.trt_model_name)
            msg = f'{self.name} load model success!'
            self.logger.info(msg)
            return msg
        except Exception as e:
            self.logger.error(e)
            return str(e)

    def unload(self, **kwargs):
        try:
            self.trt_client.unload(self.trt_model_name)
            msg = f'{self.name} unload model success!'
            self.logger.info(msg)
            return msg
        except Exception as e:
            self.logger.error(e)
            return str(e)

    def on_process(self, obj, **kwargs):
        trt_client = self.trt_client

        try:
            obj = self.request(obj, trt_client, **kwargs)
        except InferenceServerException as e:
            self.logger.error(e)
            self.logger.warning('It seem that the model is not init, reinit it!')
            trt_client = triton_utils.HttpClient(url=self.trt_url)
            obj = self.request(obj, trt_client, **kwargs)

        self.model_configs = trt_client.model_configs
        self.model_versions = trt_client.model_versions
        trt_client.client.close()

        return obj

    def request(self, obj, trt_client, **kwargs):
        """copy this template, and implement the method."""
        async_reqs = []
        for i in range(0, ..., self.batch_size):
            ...
            async_req = trt_client.async_infer(
                ...,
                model_name=self.trt_model_name
            )
            async_reqs.append(async_req)

        for i, async_req in enumerate(async_reqs):
            outputs = trt_client.async_get(async_req)
            ...

        obj.update(...)
        return obj


class LoadTritonModule(skeletons.Module):
    triton_module_name: str

    def __init__(self, *args, base_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.triton_module = base_model.get_module(self.triton_module_name)

    def on_process(self, obj, **kwargs):
        return self.triton_module.load()


class UnloadTritonModule(skeletons.Module):
    triton_module_name: str

    def __init__(self, *args, base_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.triton_module = base_model.get_module(self.triton_module_name)

    def on_process(self, obj, **kwargs):
        return self.triton_module.unload()
