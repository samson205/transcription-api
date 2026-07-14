from api.core.database import database_session
from api.core.config import settings
from api.engines.whisper_engine import WhisperEngine
from api.engines.embedding_engine import EmbeddingEngine
from api.services.transcription_service import TranscriptionService
from api.services.alignment_service import AlignmentService
from api.services.operator_service import OperatorService
from api.services.embedding_service import EmbeddingService
from api.services.speaker_match_service import SpeakerMatchService
from api.services.conversation_service import ConversationService
from api.processors.segment_aggregator import SegmentAggregator
from api.repositories.operator_repository import OperatorRepository
from api.repositories.conversation_repository import ConversationRepository
from api.orchestrators.operator_voice_orchestrator import OperatorVoiceOrchestrator
from api.orchestrators.conversation_orchestrator import ConversationOrchestrator

_WHISPER_ENGINE = None
_EMBEDDING_ENGINE = None


def get_shared_whisper_engine() -> WhisperEngine:
    global _WHISPER_ENGINE
    if _WHISPER_ENGINE is None:
        _WHISPER_ENGINE = WhisperEngine(word_timestamps=settings.WORD_TIMESTAMPS)
        _WHISPER_ENGINE._load_model()
    return _WHISPER_ENGINE


def get_shared_embedding_engine() -> EmbeddingEngine:
    global _EMBEDDING_ENGINE
    if _EMBEDDING_ENGINE is None:
        _EMBEDDING_ENGINE = EmbeddingEngine()
        _EMBEDDING_ENGINE._load_inference()
    return _EMBEDDING_ENGINE


def get_transcription_service() -> TranscriptionService:
    return TranscriptionService(get_shared_whisper_engine())


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_shared_embedding_engine())


def get_operator_repository() -> OperatorRepository:
    return OperatorRepository(database_session)


def get_operator_service() -> OperatorService:
    return OperatorService(get_operator_repository())


def get_alignment_service() -> AlignmentService:
    return AlignmentService()


def get_speaker_match_service() -> SpeakerMatchService:
    return SpeakerMatchService(get_operator_service(), get_embedding_service())


def get_segment_aggregator() -> SegmentAggregator:
    return SegmentAggregator()


def get_operator_voice_orchestrator() -> OperatorVoiceOrchestrator:
    return OperatorVoiceOrchestrator(get_operator_service(), get_embedding_service())


def get_conversation_repository() -> ConversationRepository:
    return ConversationRepository(database_session)


def get_conversation_service() -> ConversationService:
    return ConversationService(get_conversation_repository())


def get_conversation_orchestrator() -> ConversationOrchestrator:
    return ConversationOrchestrator(
        get_transcription_service(),
        get_alignment_service(),
        get_speaker_match_service(),
        get_segment_aggregator(),
        get_conversation_service(),
    )
