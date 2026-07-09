from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, ConfigDict
from fastapi import Form


class OperatorCreate(BaseModel):
    name: Annotated[str, Field(...)]

    @classmethod
    def as_form(cls, name: Annotated[str, Form(...)]):
        return cls(name=name)


class OperatorRead(BaseModel):
    name: Annotated[str, Field(...)]
    created_at: Annotated[datetime, Field(...)]

    model_config = ConfigDict(from_attributes=True)
