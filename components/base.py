from workflows import callbacks, exceptions, skeletons
from . import db, _callbacks, template
from .sdks import openai


class BaseLogPipeline(skeletons.Pipeline):
    err_response = template.BaseErrResponse

    def __init__(self, *modules, **kwargs):
        super().__init__(
            *modules,
            success_callbacks=[_callbacks.TimeLoggerCallback()],
            failure_callbacks=[_callbacks.StdErrCallback()],
            **kwargs
        )
        self.ignore_errors = True
        self.callback_wrapper.parse_exception = self.parse_exception

    def parse_exception(self, e, obj, *args, task_id=None, **kwargs) -> dict:
        if isinstance(e, exceptions._BaseException):
            ret = self.err_response(
                task_id=task_id,
                code=e.code,
                message=e.message,
            ).dict()
        else:
            ret = self.err_response(
                task_id=task_id,
                code=500,
                message=f'{type(e).__name__}: {e}',
            ).dict()
        return ret


class BaseTaskPipeline(skeletons.Pipeline):
    def gen_kwargs(self, obj, **kwargs):
        kwargs.update({k: v for k, v in obj.items() if k != 'kwargs'})
        return kwargs

    def on_process_start(self, obj, task_id=None, **kwargs):
        self.logger.info(f'{self.name}[{task_id}] receive request!')
        return obj['kwargs']

    def on_process_end(self, obj, task_id=None, **kwargs):
        return dict(
            task_id=task_id,
            data=obj
        )


class BaseModelWithoutDb(BaseLogPipeline, BaseTaskPipeline):
    pass


class BaseModelWithMysqlDb(BaseLogPipeline, BaseTaskPipeline, db.MysqlDbModule):
    """
    Attention:
        there is no error db callback module, if required, pls register by yourself or register in the module of the pipeline
    """
    query_data_first = True

    def on_process_start(self, obj, task_id=None, **kwargs):
        self.logger.info(f'{self.name}[{task_id}] receive request!')
        if self.query_data_first:
            ret = self.cacher.get_one(task_id=task_id, convert_to_json=True)
            if ret:
                pass
            else:
                _id = self.cacher.cache_one(dict(task_id=task_id))
                ret['id'] = _id

            if 'id' in ret and 'id' in obj['kwargs'] and ret['id'] != obj['kwargs']['id']:
                # request mysql again, for reprocessing history request
                ret_ = self.cacher.get_one(id=ret['id'], convert_to_json=True)
                ret.update(ret_)

        else:
            ret = {}

        ret['task_id'] = task_id
        ret.update(obj['kwargs'])
        obj['kwargs'] = ret
        return ret


class BaseCallbackModule(skeletons.Module):
    global_cacher_keys: list = []

    success_status: int = 100
    error_status: int = 500

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_callback()

    @property
    def process_info(self):
        return self.name

    def gen_kwargs(self, obj, **kwargs):
        global_caches = {k: getattr(self, k) for k in self.global_cacher_keys if hasattr(self, k)}
        general_kwargs = dict(
            global_caches=global_caches
        )
        kwargs.update(
            general_kwargs=general_kwargs
        )
        return kwargs


class MysqlCallbackModule(BaseCallbackModule):
    add_mysql_callback = True
    mysql_table: str = None
    allow_duplicates = False
    mysql_cacher_keys: list = []
    mysql_error_key: str = 'error_msg'
    mysql_status_key: str = None
    mysql_filter_mapping = {'id': 'id'}  # (mysql_key, obj_key)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.add_mysql_callback:
            self.callback_wrapper.register_success_callback(db.MysqlDbSuccessCallback(self.mysql_table))
            self.callback_wrapper.register_failure_callback(db.MySqlDbErrCallback(self.mysql_table))

    def gen_kwargs(self, obj, **kwargs):
        kwargs = super().gen_kwargs(obj, **kwargs)
        general_kwargs = kwargs.get('general_kwargs', {})
        db_general_kwargs = dict(
            mysql_table=self.mysql_table,
            allow_duplicates=self.allow_duplicates,
            filter_mapping=self.mysql_filter_mapping,
            cacher_keys=self.mysql_cacher_keys,
            error_key=self.mysql_error_key,
            status_key=self.mysql_status_key,
            success_status=self.success_status,
            error_status=self.error_status
        )
        if 'MysqlDbSuccessCallback' in kwargs['callback_status']:
            kwargs['callback_status']['MysqlDbSuccessCallback'] = dict(
                **db_general_kwargs,
                **general_kwargs
            )

        if 'MysqlDbErrCallback' in kwargs['callback_status']:
            kwargs['callback_status']['MysqlDbErrCallback'] = dict(
                **db_general_kwargs,
                **general_kwargs
            )

        return kwargs


class UrlCallbackModule(BaseCallbackModule):
    add_url_success_callback = False
    add_url_fail_callback = True
    url_cacher_keys: list = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.add_url_success_callback:
            self.callback_wrapper.register_success_callback(_callbacks.UrlSuccessCallback())
        if self.add_url_fail_callback:
            self.callback_wrapper.register_failure_callback(_callbacks.UrlErrCallback())

    def gen_kwargs(self, obj, **kwargs):
        kwargs = super().gen_kwargs(obj, **kwargs)
        general_kwargs = kwargs.get('general_kwargs', {})

        if 'UrlSuccessCallback' in kwargs['callback_status']:
            kwargs['callback_status']['UrlSuccessCallback'] = dict(
                cacher_keys=self.url_cacher_keys,
                **general_kwargs
            )

        if 'UrlErrCallback' in kwargs['callback_status']:
            kwargs['callback_status']['UrlErrCallback'] = dict(
                cacher_keys=self.url_cacher_keys,
                **general_kwargs
            )

        return kwargs


class TempMysqlCallbackModule(MysqlCallbackModule):
    add_temp_mysql_callback = True
    temp_mysql_table: str
    temp_mysql_global_cacher_keys: list = []
    temp_mysql_cacher_keys: list = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.add_temp_mysql_callback:
            self.callback_wrapper.register_success_callback(db.MysqlDbSuccessCallback(self.temp_mysql_table, name='TempMysqlDbSuccessCallback'))

    def gen_kwargs(self, obj, **kwargs):
        kwargs = super().gen_kwargs(obj, **kwargs)
        if 'TempMysqlDbSuccessCallback' in kwargs['callback_status']:
            temp_global_caches = {k: getattr(self, k) for k in self.temp_mysql_global_cacher_keys if hasattr(self, k)}
            kwargs['callback_status']['TempMysqlDbSuccessCallback'] = dict(
                allow_duplicates=self.allow_duplicates,
                filter_mapping=self.mysql_filter_mapping,
                mysql_table=self.temp_mysql_table,
                global_caches=temp_global_caches,
                cacher_keys=self.temp_mysql_cacher_keys,
            )

        return kwargs


class OpenaiMysqlCallbackModule(openai.Openai, MysqlCallbackModule):
    add_url_callback = False

    mysql_cacher_keys: list = ['model', 'messages', 'content', 'reasoning_content', 'total_tokens', 'prompt_tokens', 'completion_tokens', 'reasoning_tokens']
    mysql_filter_mapping = {'pid': 'pid', 'sid': 'sid'}  # (mysql_key, obj_key)

    global_cacher_keys = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.callback_wrapper.register_success_callback(
            callbacks.TimeLoggerCallback(
                fmt='[{pid}-{sid}] request ' + self.model + ' takes {time:.2f} s'
            )
        )

        self.callback_wrapper.register_success_callback(
            _callbacks.TimeDbCacheCallback(
                mysql_table=self.mysql_table,
                db_filter_mapping=self.mysql_filter_mapping,
                cache_key='duration'
            )
        )
        self.ignore_errors = False

    def gen_kwargs(self, obj, **kwargs):
        kwargs = super().gen_kwargs(obj, **kwargs)
        kwargs.setdefault('sid', None)
        return kwargs

    def on_process_start(self, obj, pid=None, sid=None, **kwargs):
        obj.update(
            pid=pid,
            sid=sid,
            model=self.model,
        )
        return obj

    def on_process_end(self, obj, **kwargs):
        post_kwargs = obj['post_kwargs']
        post_result = obj['post_result']

        messages = post_kwargs['messages']

        message = post_result.choices[0].message
        content = message.content
        reasoning_content = message.model_extra.get('reasoning_content', '')

        usage = post_result.usage
        completion_tokens = usage.completion_tokens
        prompt_tokens = usage.prompt_tokens
        total_tokens = usage.total_tokens
        reasoning_tokens = usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details else 0

        obj.update(
            messages=messages,
            content=content,
            reasoning_content=reasoning_content,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
        )

        return obj


class VolcengineMysqlModule(openai.Volcengine):
    """
    Usage:
        class LlmRequest(VolcengineDbCallbackModule):
            def on_process(self, obj, task_id=None, **kwargs):
                ...
                llm_result = self.request(
                    sys,
                    user,
                    global_kwargs=dict(
                        pid=obj['id'],
                        task_id=task_id,
                        sid="xxx"
                    )
                )
                ...
                return obj

    """
    llm_cacher_mysql_table: str
    max_input_length: int = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_client = OpenaiMysqlCallbackModule(
            model=self.model,
            client_kwargs=self.client_kwargs,
            mysql_table=self.llm_cacher_mysql_table,
            max_input_length=self.max_input_length
        )
