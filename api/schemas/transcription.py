import uuid
from typing import Annotated
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict
from faster_whisper.transcribe import Segment, Word

from api.models.enums import ProcessingStatus


class SegmentSchema(BaseModel):
    start: Annotated[float, Field(...)]
    end: Annotated[float, Field(...)]
    text: Annotated[str, Field(...)]


class RawTranscriptionSchema(BaseModel):
    language: Annotated[str, Field(...)]
    duration: Annotated[float, Field(...)]
    segments: Annotated[list[Segment | Word], Field(...)]


class SpeakerSegment(BaseModel):
    start: Annotated[float, Field(...)]
    end: Annotated[float, Field(...)]
    speaker: Annotated[str, Field(...)]


class DialogueSegment(SpeakerSegment):
    text: Annotated[str, Field(...)]


class ConversationResponse(BaseModel):
    id: Annotated[int, Field(...)]
    filename: Annotated[str, Field(...)]
    status: Annotated[ProcessingStatus, Field(...)]
    error_message: Annotated[str | None, Field(None)]
    language: Annotated[str | None, Field(None)]
    duration: Annotated[float | None, Field(None)]
    created_at: Annotated[datetime, Field(...)]
    segments: Annotated[list[DialogueSegment] | None, Field(None)]

    model_config = ConfigDict(from_attributes=True)
