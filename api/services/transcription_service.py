import logging
import time

from api.schemas.transcription import RawTranscriptionSchema
from api.engines.whisper_engine import WhisperEngine

logger = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(self, whisper: WhisperEngine) -> None:
        self._whisper = whisper

    def transcribe_file(self, path: str):
        segments, info = self._whisper.transcribe(path)

        start = time.monotonic()
        data_to_return = []
        for s in segments:
            if s.words:
                data_to_return.extend(s.words)
            else:
                data_to_return.append(s)
        elapsed = time.monotonic() - start

        logger.info(
            "Transcription done file=%s, lang=%s, duration=%.2fs, segments=%d, took=%.2fs",
            path,
            info.language,
            info.duration,
            len(data_to_return),
            elapsed,
        )
        return RawTranscriptionSchema(
            language=info.language,
            duration=round(info.duration, 1),
            segments=data_to_return,
        )
