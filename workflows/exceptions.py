class _BaseException(Exception):
    message: str
    code: int

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self.message


class ServerException(_BaseException):
    message = 'Unknown error.'
    code = 500


class EarlyStopException(_BaseException):
    message = 'Service not completed.'
    code = 100


class NullDataException(_BaseException):
    message = 'Data is empty.'
    code = 400


class InputMissingException(_BaseException):
    message = 'Input field is missing.'
    code = 401

    def __init__(self, keys=[], reason='Missing fields', **kwargs):
        super().__init__(**kwargs)
        if keys:
            s = ''
            for key in keys:
                s += f'`{key}`, '
            s = s[:-2]
            self.message += f' {reason}: {s}'


class InputInvalidException(_BaseException):
    message = 'Invalid input data.'
    code = 402

    def __init__(self, keys='', reason='Invalid fields', **kwargs):
        super().__init__(**kwargs)
        if keys:
            if isinstance(keys, str):
                keys = [keys]
            s = ''
            for key in keys:
                s += f'`{key}`, '
            s = s[:-2]
            self.message += f' {reason}: {s}'


class FileDownloadException(_BaseException):
    message = 'File download failed.'
    code = 403

    def __init__(self, url, **kwargs):
        super().__init__(**kwargs)
        self.message += f' Download URL: {url}'


class FileNotFoundException(_BaseException):
    message = 'File not found.'
    code = 404

    def __init__(self, fp, **kwargs):
        super().__init__(**kwargs)
        self.message += f' File path: {fp}'


class TimeOutException(_BaseException):
    message = 'Request timed out.'
    code = 501


class LLMInputOutOfLengthException(_BaseException):
    code = 2403

    def __init__(self, length: int, max_length: int, **kwargs):
        self.message = f'Input length out of limit, max length is {max_length}, current length is {length}!'
        super().__init__(**kwargs)


class LLMBlockException(_BaseException):
    message = 'The content has triggered safety review rules. This request cannot be completed. Please adjust the content and try again.'
    code = 2404

    def __init__(self, block_content: str = None, **kwargs):
        super().__init__(**kwargs)
        if block_content:
            self.message += f' Blocked words: {block_content}'


class LLMParseException(_BaseException):
    message = 'Failed to parse LLM response.'
    code = 2405

    def __init__(self, llm_result: str = None, reason='LLM response', **kwargs):
        super().__init__(**kwargs)
        if llm_result:
            self.message += f' {reason}:\n{llm_result}'
