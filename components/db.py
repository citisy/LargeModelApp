import os

import pymysql

from utils import os_lib, op_utils
from workflows import callbacks, skeletons


class MysqlDbModule(skeletons.Module):
    mysql_host = os.getenv('mysql_host', 'xxx')
    mysql_port = os.getenv('mysql_port', 'xxx')
    mysql_user = os.getenv('mysql_user', 'xxx')
    mysql_pwd = os.getenv('mysql_pwd', 'xxx')
    mysql_db = os.getenv('mysql_db', 'xxx')
    mysql_table: str = ''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cacher = os_lib.MySqlCacher(
            host=self.mysql_host,
            port=self.mysql_port,
            user=self.mysql_user,
            password=self.mysql_pwd,
            database=self.mysql_db,
            table=self.mysql_table
        )
        self.retry = op_utils.Retry(stdout_method=self.logger.info, count=3, wait=5)
        self.cacher.get_one = self.retry.add_try(err_type=pymysql.err.OperationalError)(self.cacher.get_one)
        self.cacher.cache_one = self.retry.add_try(err_type=pymysql.err.OperationalError)(self.cacher.cache_one)


class UpdateMysqlDbModule(MysqlDbModule):
    cacher_keys: list
    process_info: str
    task_process: int
    allow_duplicates = False
    filter_mapping = {'id': 'id'}  # (database_key, obj_key)
    pri_key = 'id'

    def on_process_end(self, obj: dict, debug=False, **kwargs):
        data = dict()

        if hasattr(self, 'task_process'):
            data['task_process'] = self.task_process

        if hasattr(self, 'process_info'):
            data['process_info'] = self.process_info

        for k in self.cacher_keys:
            if k in obj and obj[k] is not None:
                data[k] = obj[k]

        cache_kwargs = dict(
            allow_duplicates=self.allow_duplicates,
            pri_key=self.pri_key
        )
        for database_key, obj_key in self.filter_mapping.items():
            if obj_key in obj:
                cache_kwargs[database_key] = obj[obj_key]

        _id = self.cacher.cache_one(data, **cache_kwargs)
        obj.setdefault(self.pri_key, _id)

        return obj


class FakeCacher:
    def get_one(self, *args, **kwargs):
        return {}

    def cache_one(self, *args, **kwargs):
        pass


class MysqlDbSuccessCallback(callbacks.Module):
    def __init__(self, mysql_table: str, **kwargs):
        super().__init__(**kwargs)
        if mysql_table:
            self.cacher = MysqlDbModule(
                mysql_table=mysql_table
            ).cacher
        else:
            self.cacher = FakeCacher()

    def on_process(
            self, obj,
            callback_status=dict(),
            **kwargs
    ) -> None:
        status = callback_status[self.name]

        data = dict()

        status_key = status.get('status_key')
        if status_key:
            data[status_key] = status['success_status']

        error_key = status.get('error_key')
        if error_key:
            data[error_key] = ''

        global_caches = status['global_caches']
        data.update(global_caches)

        cacher_keys = status['cacher_keys']
        for k in cacher_keys:
            if k in obj and obj[k] is not None:
                data[k] = obj[k]

        cache_kwargs = dict(
            allow_duplicates=status['allow_duplicates']
        )
        for database_key, obj_key in status['filter_mapping'].items():
            if obj_key in obj:
                cache_kwargs[database_key] = obj[obj_key]

        if data:
            self.cacher.cache_one(data, **cache_kwargs)


class MySqlDbErrCallback(callbacks.Module):
    def __init__(self, mysql_table: str, **kwargs):
        super().__init__(**kwargs)
        if mysql_table:
            self.cacher = MysqlDbModule(
                mysql_table=mysql_table
            ).cacher
        else:
            self.cacher = FakeCacher()

    def on_process(
            self,
            obj, parse_obj=None, e=None,
            callback_status=dict(),
            **kwargs
    ) -> None:
        status = callback_status[self.name]

        data = {}

        error_key = status.get('error_key')
        if error_key:
            if isinstance(parse_obj, dict):
                error_msg = parse_obj.get('message', str(e))
            elif hasattr(parse_obj, 'message'):
                error_msg = parse_obj.message
            else:
                error_msg = f'{type(e).__name__}: {e}'

            data[error_key] = error_msg

        status_key = status.get('status_key')
        if status_key is not None:
            if hasattr(parse_obj, 'status'):
                data[status_key] = parse_obj.status
            elif hasattr(parse_obj, 'code'):
                data[status_key] = parse_obj.code
            else:
                data[status_key] = status['error_status']

        global_caches = status['global_caches']
        data.update(global_caches)

        cacher_keys = status['cacher_keys']
        for k in cacher_keys:
            if k in obj and obj[k] is not None:
                data[k] = obj[k]

        cache_kwargs = dict(
            allow_duplicates=False
        )
        for database_key, obj_key in status['filter_mapping'].items():
            if obj_key in obj:
                cache_kwargs[database_key] = obj[obj_key]

        self.cacher.cache_one(
            data,
            **cache_kwargs
        )
