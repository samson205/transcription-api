from typing import Annotated, Any
from datetime import datetime

from fastapi import Form
from pydantic import BaseModel, Field, ConfigDict


class SegmentSchema(BaseModel):
    start: Annotated[float, Field(...)]
    end: Annotated[float, Field(...)]
    text: Annotated[str, Field(...)]


class TranscriptionSchema(BaseModel):
    language: Annotated[str, Field(...)]
    duration: Annotated[float, Field(...)]
    segments: Annotated[list[SegmentSchema], Field(...)]


class SpeakerSegment(BaseModel):
    start: Annotated[float, Field(...)]
    end: Annotated[float, Field(...)]
    speaker: Annotated[str, Field(...)]


class BaseTaskResponse(BaseModel):
    task_id: Annotated[str, Field(...)]


class DialogueSegment(SpeakerSegment):
    text: Annotated[str, Field(...)]
    

class ConversationResponse(BaseModel):
    language: Annotated[str, Field(...)]
    duration: Annotated[float, Field(...)]
    segments: Annotated[list[DialogueSegment], Field(...)]


class TaskResponse(BaseTaskResponse):
    task_id: Annotated[str, Field(...)]
    state: Annotated[str, Field(...)]
    result: Annotated[Any | None, Field(...)]


class OperatorCreate(BaseModel):
    name: Annotated[str, Field(...)]

    @classmethod
    def as_form(cls, name: Annotated[str, Form(...)]):
        return cls(name=name)
    

class OperatorRead(BaseModel):
    name: Annotated[str, Field(...)]
    created_at: Annotated[datetime, Field(...)]

    model_config = ConfigDict(from_attributes=True)
