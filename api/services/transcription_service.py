from typing import Iterable

from faster_whisper.transcribe import Segment

from api.schemas.transcription import TranscriptionSchema
from api.engines.whisper_engine import WhisperEngine


class TranscriptionService:
    def __init__(self, whisper: WhisperEngine) -> None:
        self._whisper = whisper

    def transcribe_file(self, path: str):
        segments, info = self._whisper.transcribe(path)
        rebuild_segments = self._rebuild_segments(segments)
        
        return TranscriptionSchema(
            language=info.language,
            duration=round(info.duration, 1),
            segments=rebuild_segments
        )
    
    @staticmethod
    def _rebuild_segments(segments: Iterable[Segment]):
        result = []
        current = None

        for segment in segments:
            if current is None:
                current = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                }
            else:
                current["end"] = segment.end
                current["text"] += " " + segment.text.strip()

            if current["text"].endswith((".", "!", "?")):
                result.append(current)
                current = None

        if current:
            result.append(current)
        
        return result
