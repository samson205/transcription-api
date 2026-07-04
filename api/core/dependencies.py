from fastapi import Depends

from api.transcription.service import TranscriptionService
from api.services.whisper_engine import WhisperEngine

_whisper_engine = WhisperEngine()


def get_whisper_engine() -> WhisperEngine:
    return _whisper_engine


def get_transcription_service(whisper: WhisperEngine = Depends(get_whisper_engine)) -> TranscriptionService:
    return TranscriptionService(whisper)
