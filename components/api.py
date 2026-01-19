import os
import time

import fastapi  # noqa

from utils import web_app, log_utils, converter
from workflows import skeletons


def add_process_time_header(app, **kwargs):
    @app.middleware("http")
    async def add(request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f'{process_time:0.4f} sec'
        return response


def add_middleware(app, **kwargs):
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def add_webui(
        app,
        model_configs=dict(),
        webui_app_func=None,
        sub_app=None,
        router_path=None,
        api_configs=None,
        **router_kwargs
):
    import gradio as gr

    if isinstance(webui_app_func, str):
        webui_app_func = converter.DataInsConvert.str_to_instance(webui_app_func)
    webui_app = webui_app_func(model_configs)
    gr.mount_gradio_app(app, webui_app, **router_kwargs)


def add_ht(app, path):
    @app.get(f"{path}/ht")
    async def health_check():
        return app.version


def add_info_api(app, path, config):
    info = {}
    for k1, v1 in config.items():
        for k2, _config in v1.items():
            if _config['model']:
                info[f'{k1}{k2}'] = _config['model'].module_info()

    @app.get(f"{path}/infos")
    async def infos():
        return info


def add_docs(
        app, router_path, api_path,
        title='',
        version="1.0.0",
        description=None,
        **router_kwargs
):
    from fastapi.openapi.docs import get_swagger_ui_html
    from fastapi import HTTPException
    from fastapi.openapi.utils import get_openapi
    from fastapi.responses import FileResponse

    @app.get(f"/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=f'{router_path}/openapi.json',
            title=title + " - Swagger UI",
            swagger_js_url=f"{router_path}/static/swagger-ui-bundle.js",
            swagger_css_url=f"{router_path}/static/swagger-ui.css",
            swagger_favicon_url=f"{router_path}/static/favicon-32x32.png",
        )

    @app.get("/static/{file_path:path}", include_in_schema=False)
    async def get_static_file(file_path: str):
        file_location = os.path.join('static', file_path)

        if not os.path.exists(file_location) or not os.path.isfile(file_location):
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(file_location)

    @app.get("/openapi.json", include_in_schema=False)
    async def custom_openapi():
        openapi_schema = get_openapi(
            title=title,
            version=version,
            routes=app.routes,
            description=description
        )
        openapi_schema['paths'] = {router_path + k: v for k, v in openapi_schema['paths'].items()}
        return openapi_schema


class BaseServer(skeletons.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._modules = [
            self.on_receive_start,
            *self._nodes,
            self.on_respond_end
        ]

    def on_receive_start(self, obj, **kwargs):
        return obj

    def on_respond_end(self, obj, **kwargs):
        return obj


class AsyncServer(BaseServer):
    def __init__(self, model, n_pool=5, logger=None):
        super().__init__()
        from concurrent.futures import ThreadPoolExecutor

        self.model = model
        self.pool = ThreadPoolExecutor(max_workers=n_pool)
        self.logger = logger
        self.ignore_errors = True

    def on_process(self, data, **kwargs):
        self.logger.info(f'Get request[{data["task_id"]}]')
        self.pool.submit(self.model, data, **kwargs)
        return {'task_id': data['task_id']}


class SyncServer(BaseServer):
    def __init__(self, model, logger=None):
        super().__init__()
        self.model = model
        self.logger = logger

    def on_process(self, data, **kwargs):
        r = self.model(data, **kwargs)

        return r


def simple_get_router(
        app: 'FastAPI' or 'APIRouter',
        router_path,
        api_path,
        func=None,
        func_configs: dict = {},
        method_configs: dict = {},
        **ignore_kwargs
):
    @app.get(api_path, **method_configs)
    def get():
        ret = func(None, **func_configs)
        return ret


def create_app(configs):
    log_configs = configs.get('Log', {})
    log_utils.logger_init(**log_configs)
    logger = log_utils.get_logger()

    api_configs = configs.get('Api', {})

    _api_configs = {}
    model_mapping = {}
    for k1, v1 in api_configs.items():
        tmp = {}
        for k2, _config in v1.items():
            if not _config.get('apply', True):
                continue

            cfgs = _config.get('model_configs') or {}
            additional_configs = {}
            if 'base_model' in cfgs and cfgs['base_model'] in model_mapping:
                additional_configs.update(
                    base_model=model_mapping[cfgs['base_model']]
                )

            model_instance = _config.get('model_instance', None)
            if model_instance:
                model = converter.DataInsConvert.str_to_instance(_config['model_instance']).from_configs(cfgs, logger=logger, **additional_configs)
                model_mapping[k1 + k2] = model
                logger.info(f'Model init:\n{model}')
            else:
                model = None

            request_template = _config.get('request_template', None)
            if request_template:
                request_template = converter.DataInsConvert.str_to_instance(_config['request_template'])
            response_template = _config.get('response_template', None)
            if response_template:
                response_template = converter.DataInsConvert.str_to_instance(_config['response_template'])

            _config.update(
                model=model,
                request_template=request_template,
                response_template=response_template,
            )

            add_sync = _config.get('add_sync', False)
            if add_sync:
                p = f'{k2}/sync'

                tmp[p] = dict(
                    func=SyncServer(model, logger=logger),
                    **_config
                )
                logger.info(f'Api init: {k1}{p}')

            add_async = _config.get('add_async', False)
            if add_async:
                p = f'{k2}/async'
                tmp[p] = dict(
                    func=AsyncServer(model, n_pool=_config.get('num_async_worker', 5), logger=logger),
                    **_config
                )
                logger.info(f'Api init: {k1}{p}')

            if not (add_sync or add_async):
                p = k2
                tmp[p] = dict(
                    func=model,
                    **_config
                )
                logger.info(f'Api init: {k1}{p}')

        if tmp:
            _api_configs[k1] = tmp

    app_configs = configs.get('App', {})
    router_configs = configs.get('Router', {})
    router_configs['api_configs'] = _api_configs

    app = web_app.FastapiOp.from_configs(app_configs=app_configs, router_configs=router_configs)

    return app
