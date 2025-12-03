import time

import requests

from workflows import callbacks, exceptions
from . import db, template


class TimeLoggerCallback(callbacks.TimeLoggerCallback):
    def __init__(self, **kwargs):
        fmt = '{module_name}[{task_id}] takes {time:.2f} s!'
        super().__init__(fmt=fmt, **kwargs)


class TimeDbCacheCallback(callbacks.TimeLoggerCallback):
    cache_key: str
    db_filter_mapping = {'id': 'id'}  # (mysql_key, obj_key)

    def __init__(self, mysql_table, **kwargs):
        fmt = '{module_name}[{task_id}] takes {time:.2f} s!'
        super().__init__(fmt=fmt, **kwargs)
        self.cacher = db.MysqlDbModule(
            mysql_table=mysql_table
        ).cacher

    def on_process_end(self, obj, callback_status={}, **kwargs) -> None:
        st = callback_status[self.name]
        et = time.time()
        t = et - st
        update_obj = {
            self.cache_key: t
        }

        cacher_kwargs = dict(
            allow_duplicates=False
        )
        for k, v in self.db_filter_mapping.items():
            if v in obj:
                cacher_kwargs[k] = obj[v]
            elif 'data' in obj and v in obj['data']:
                cacher_kwargs[k] = obj['data'][v]

        self.cacher.cache_one(update_obj, **cacher_kwargs)


class StdErrCallback(callbacks.StdErrCallback):
    def __init__(self, **kwargs):
        fmt = '{module_name}[{task_id}] encounters errors:\n{tb_info}'
        super().__init__(fmt=fmt, **kwargs)


class UrlSuccessCallback(callbacks.Module):
    success_template = template.BaseSuccessResponse

    def on_process(
            self, obj,
            task_id=None, callback_url=None,
            callback_status=dict(),
            **kwargs
    ):
        if not callback_url:
            return

        status = callback_status[self.name]
        cacher_keys = status['cacher_keys']

        data = dict()

        for k in cacher_keys:
            if k in obj and obj[k] is not None:
                data[k] = obj[k]

        global_caches = status['global_caches']

        ret = self.success_template(
            task_id=task_id,
            data=data,
            **global_caches
        ).dict()

        r = requests.post(callback_url, json=ret)
        r.raise_for_status()
        return r


class UrlErrCallback(callbacks.Module):
    err_response = template.BaseErrResponse

    def on_process(
            self, obj,
            e=None,
            task_id=None, callback_url=None,
            callback_status=dict(), **kwargs
    ):
        if not callback_url:
            return

        status = callback_status[self.name]
        global_caches = status['global_caches']

        if isinstance(e, exceptions._BaseException):
            ret = self.err_response(
                task_id=task_id,
                code=e.code,
                message=e.message,
                **global_caches
            ).dict()
        else:
            ret = self.err_response(
                task_id=task_id,
                code=500,
                message=f'{type(e).__name__}: {e}',
                **global_caches
            ).dict()
        r = requests.post(callback_url, json=ret)
        r.raise_for_status()
        return r
