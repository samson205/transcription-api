from api.schemas.transcription import RawTranscriptionSchema
from api.engines.whisper_engine import WhisperEngine


class TranscriptionService:
    def __init__(self, whisper: WhisperEngine) -> None:
        self._whisper = whisper

    def transcribe_file(self, path: str):
        segments, info = self._whisper.transcribe(path)

        return RawTranscriptionSchema(
            language=info.language,
            duration=round(info.duration, 1),
            segments=segments,
        )
