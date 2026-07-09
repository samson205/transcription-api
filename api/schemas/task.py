from typing import Annotated, Any

from pydantic import BaseModel, Field


class BaseTaskResponse(BaseModel):
    task_id: Annotated[str, Field(...)]


class TaskResponse(BaseTaskResponse):
    task_id: Annotated[str, Field(...)]
    state: Annotated[str, Field(...)]
    result: Annotated[Any | None, Field(...)]
