from pathlib import Path

from api.schemas.schemas import Transcription
from api.engines.whisper import WhisperEngine
from api.engines.diarization import DiarizationEngine
from api.services.segments import SegmentsService


class TranscriptionService:
    def __init__(self, whisper: WhisperEngine, diarization: DiarizationEngine) -> None:
        self._whisper = whisper
        self._diarization = diarization

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
