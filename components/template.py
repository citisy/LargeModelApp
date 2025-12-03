import uuid
from typing import List

from pydantic import BaseModel, Field

new_id = lambda: ''.join(str(uuid.uuid4()).split('-'))[:8]


class BaseRequest(BaseModel):
    task_id: str = Field(default_factory=new_id)
    callback_url: str = None
    debug: bool = False
    mask_modules: List = []
    apply_modules: List = []
    start_module: str = None
    end_module: str = None


class BaseSuccessResponse(BaseModel):
    task_id: str | int
    code: int = 200
    message: str = 'success'
    data: dict = {}


class BaseErrResponse(BaseModel):
    task_id: str
    code: int
    message: str
    data: dict = {}
