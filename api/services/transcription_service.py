from api.schemas.transcription import RawTranscriptionSchema
from api.engines.whisper_engine import WhisperEngine


class TranscriptionService:
    def __init__(self, whisper: WhisperEngine) -> None:
        self._whisper = whisper

    def transcribe_file(self, path: str):
        segments, info = self._whisper.transcribe(path)

        data_to_return = []
        for s in segments:
            if s.words:
                data_to_return.extend(s.words)
            else:
                data_to_return.append(s)

        return RawTranscriptionSchema(
            language=info.language,
            duration=round(info.duration, 1),
            segments=data_to_return,
        )
