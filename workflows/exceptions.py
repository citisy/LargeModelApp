class _BaseException(Exception):
    message: str
    code: int

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ServerException(_BaseException):
    code = 500


class EarlyStopException(_BaseException):
    code = 100


class LLMInputOutOfLengthException(_BaseException):
    code = 2403

    def __init__(self, length: int, max_length: int, **kwargs):
        self.message = f'Input length out of limit, max length is {max_length}, current length is {length}'
        super().__init__(**kwargs)


class LLMBlockException(_BaseException):
    message = 'Input contains blocked words'
    code = 2404
