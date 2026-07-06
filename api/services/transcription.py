from pathlib import Path

from api.core.dependencies import get_diarization_engine, get_whisper_engine
from api.schemas.schemas import Transcription
from api.services.segments import SegmentsService


class TranscriptionService:
    def __init__(self) -> None:
        self._whisper = get_whisper_engine()
        self._diarization = get_diarization_engine()

    def transcribe_file(self, path: str):
        try:
            segments, info = self._whisper.transcribe(path)
            self._diarization.diarize_audio(path)
        finally:
            Path(path).unlink(missing_ok=True)
        rebuild_segments = SegmentsService.rebuild_segments(segments)
        
        return Transcription(
            language=info.language,
            duration=round(info.duration, 1),
            segments=rebuild_segments
        )
