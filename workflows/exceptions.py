class _BaseException(Exception):
    message: str
    code: int

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ServerException(_BaseException):
    code = 500


class EarlyStopException(_BaseException):
    code = 100
