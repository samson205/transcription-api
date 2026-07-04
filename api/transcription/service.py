from api.services.whisper_engine import WhisperEngine


class TranscriptionService:
    def __init__(self, whisper: WhisperEngine) -> None:
        self._whisper = whisper

    def get_transcription(self, path: str):
        segments, info = self._whisper.transcribe(path)
        return {
        "language": info.language,
        "duration": round(info.duration, 1),
        "segments": [
            {
                "start": round(segment.start, 1),
                "end": round(segment.end, 1),
                "text": segment.text.strip()
            }
            for segment in segments
        ]
    }
