from typing import Annotated, Iterable

from pydantic import BaseModel, Field
from faster_whisper.transcribe import Segment


class SegmentSchema(BaseModel):
    start: Annotated[float, Field(...)]
    end: Annotated[float, Field(...)]
    text: Annotated[str, Field(...)]


class RawTranscriptionSchema(BaseModel):
    language: Annotated[str, Field(...)]
    duration: Annotated[float, Field(...)]
    segments: Annotated[Iterable[Segment], Field(...)]


class SpeakerSegment(BaseModel):
    start: Annotated[float, Field(...)]
    end: Annotated[float, Field(...)]
    speaker: Annotated[str, Field(...)]


class DialogueSegment(SpeakerSegment):
    text: Annotated[str, Field(...)]


class ConversationResponse(BaseModel):
    language: Annotated[str, Field(...)]
    duration: Annotated[float, Field(...)]
    segments: Annotated[list[DialogueSegment], Field(...)]
