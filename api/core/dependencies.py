from api.services.transcription import TranscriptionService
from api.engines.whisper import WhisperEngine
from api.engines.diarization import DiarizationEngine

_whisper_engine = WhisperEngine()
_diarization_engine = DiarizationEngine()
_transcription_service = TranscriptionService(_whisper_engine, _diarization_engine)


def get_whisper_engine() -> WhisperEngine:
    return _whisper_engine


def get_diarization_engine() -> DiarizationEngine:
    return _diarization_engine


def get_transcription_service() -> TranscriptionService:
    return _transcription_service

