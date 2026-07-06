from functools import lru_cache

from api.engines.whisper_engine import WhisperEngine
from api.engines.diarization_engine import DiarizationEngine
from api.services.transcription_service import TranscriptionService
from api.services.diarization_service import DiarizationService
from api.services.alignment_service import AlignmentService
from api.services.conversation_service import ConversationService


@lru_cache
def get_whisper_engine() -> WhisperEngine:
    return WhisperEngine()


@lru_cache
def get_diarization_engine() -> DiarizationEngine:
    return DiarizationEngine()


@lru_cache
def get_transcription_service() -> TranscriptionService:
    return TranscriptionService(get_whisper_engine())


@lru_cache
def get_diarization_service() -> DiarizationService:
    return DiarizationService(get_diarization_engine())


@lru_cache
def get_alignment_service() -> AlignmentService:
    return AlignmentService()


@lru_cache
def get_conversation_service() -> ConversationService:
    return ConversationService(
        get_transcription_service(),
        get_diarization_service(),
        get_alignment_service(),
    )
