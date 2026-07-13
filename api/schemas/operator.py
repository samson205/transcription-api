from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, ConfigDict
from fastapi import Form

from api.models.enums import ProcessingStatus


class OperatorCreate(BaseModel):
    name: Annotated[str, Field(...)]

    @classmethod
    def as_form(cls, name: Annotated[str, Form(...)]):
        return cls(name=name)


class OperatorRead(BaseModel):
    id: Annotated[int, Field(...)]
    name: Annotated[str, Field(...)]
    status: Annotated[ProcessingStatus, Field(...)]
    error_message: Annotated[str | None, Field(None)]
    created_at: Annotated[datetime, Field(...)]

    model_config = ConfigDict(from_attributes=True)
