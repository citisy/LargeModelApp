class ServerException(Exception):
    def __init__(self, message, code=500):
        self.message = message
        self.code = code


class EarlyStopException(Exception):
    def __init__(self, message, code=100):
        self.message = message
        self.code = code
