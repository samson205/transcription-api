from typing import Annotated

from pydantic import BaseModel, Field


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
    result: Annotated[ConversationResponse | None, Field(...)]
