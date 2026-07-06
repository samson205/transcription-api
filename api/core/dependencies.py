from api.engines.whisper import WhisperEngine
from api.engines.diarization import DiarizationEngine

_whisper_engine = None
_diarization_engine = None


def get_whisper_engine() -> WhisperEngine:
    global _whisper_engine
    if _whisper_engine is None:
        _whisper_engine = WhisperEngine()
    return _whisper_engine


def get_diarization_engine() -> DiarizationEngine:
    global _diarization_engine
    if _diarization_engine is None:
        _diarization_engine = DiarizationEngine()
    return _diarization_engine
