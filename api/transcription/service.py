from pathlib import Path

from fastapi import UploadFile

from api.transcription.schemas import Transcription
from api.services.whisper_engine import WhisperEngine
from api.services.segments_service import SegmentsService
from api.services.temp_storage import TempStorage


class TranscriptionService:
    def __init__(self, whisper: WhisperEngine) -> None:
        self._whisper = whisper

    def transcribe_file(self, file: UploadFile):
        tmp_path = TempStorage.get_temp_file(file)
        try:
            segments, info = self._whisper.transcribe(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
        rebuild_segments = SegmentsService.rebuild_segments(segments)
        
        return Transcription(
            language=info.language,
            duration=round(info.duration, 1),
            segments=rebuild_segments
        )
