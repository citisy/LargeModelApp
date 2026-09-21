class _BaseException(Exception):
    message: str
    code: int

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self.message


class EarlyStopException(_BaseException):
    message = 'Service not completed.'
    code = 100


class InputException(_BaseException):
    message = 'Unknown input error.'
    code = 400


class InputKwargsException(_BaseException):
    message = 'Some wrong with input kwargs.'
    code = 410


class InputMissingException(_BaseException):
    message = 'Input field is missing.'
    code = 411

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
    code = 412

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


class InputDataException(_BaseException):
    message = 'Some wrong with input data.'
    code = 420


class NullDataException(_BaseException):
    message = 'Data is empty.'
    code = 421


class InputUrlException(_BaseException):
    message = 'Some wrong with input url.'
    code = 430


class FileDownloadException(_BaseException):
    message = 'File download failed.'
    code = 431

    def __init__(self, url, **kwargs):
        super().__init__(**kwargs)
        self.message += f' Download URL: {url}'


class FileNotFoundException(_BaseException):
    message = 'File not found.'
    code = 441

    def __init__(self, fp, **kwargs):
        super().__init__(**kwargs)
        self.message += f' File path: {fp}'


class ServerException(_BaseException):
    message = 'Unknown error with server processing.'
    code = 500


class TimeOutException(_BaseException):
    message = 'Processing timed out.'
    code = 501


class OutputKwargsException(_BaseException):
    message = 'Some wrong with output kwargs.'
    code = 510


class OutputDataException(_BaseException):
    message = 'Some wrong with output data.'
    code = 520


class TextProcessException(_BaseException):
    message = 'Text process unknown error.'
    code = 1500


class ImageInvalidException(_BaseException):
    message = 'No valid image content detected or input image is empty.'
    code = 2412


class ImageTooSmallException(_BaseException):
    message = 'Too small valid image content detected.'
    code = 2413


class ImageProcessException(_BaseException):
    message = 'Image process unknown error.'
    code = 2500


class AudioInvalidException(_BaseException):
    message = 'No valid audio content detected or input audio is empty.'
    code = 3412


class AudioTooShortException(_BaseException):
    message = 'Too short valid audio content detected.'
    code = 3413


class AudioProcessException(_BaseException):
    message = 'Audio process unknown error.'
    code = 3500


class VideoInvalidException(_BaseException):
    message = 'No valid video content detected or input video is empty.'
    code = 4412


class VideoTooShortException(_BaseException):
    message = 'Too short valid video content detected.'
    code = 4413


class VideoProcessException(_BaseException):
    message = 'Video process unknown error.'
    code = 4500


class LLMProcessException(_BaseException):
    message = 'LLM process unknown error.'
    code = 5500


class LLMInputOutOfLengthException(_BaseException):
    code = 5414

    def __init__(self, length: int, max_length: int, **kwargs):
        self.message = f'Input length out of limit, max length is {max_length}, current length is {length}!'
        super().__init__(**kwargs)


class LLMBlockException(_BaseException):
    message = 'The content has triggered safety review rules. This request cannot be completed. Please adjust the content and try again.'
    code = 5422

    def __init__(self, block_content: str = None, **kwargs):
        super().__init__(**kwargs)
        if block_content:
            self.message += f' Blocked words: {block_content}'


class LLMParseException(_BaseException):
    message = 'Failed to parse LLM response.'
    code = 5521

    def __init__(self, llm_result: str = None, reason='LLM response', **kwargs):
        super().__init__(**kwargs)
        if llm_result:
            self.message += f' {reason}:\n{llm_result}'
