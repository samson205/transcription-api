from fastapi import Depends

from api.transcription.service import TranscriptionService
from api.services.whisper_engine import WhisperEngine
from api.services.diarization_engine import DiarizationEngine

_whisper_engine = WhisperEngine()
_diarization_engine = DiarizationEngine()


def get_whisper_engine() -> WhisperEngine:
    return _whisper_engine


def get_diarization_engine() -> DiarizationEngine:
    return _diarization_engine


def get_transcription_service(
    whisper: WhisperEngine = Depends(get_whisper_engine),
    diarization: DiarizationEngine = Depends(get_diarization_engine)
) -> TranscriptionService:
    return TranscriptionService(whisper, diarization)
