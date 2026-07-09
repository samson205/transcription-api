from api.core.database import database_session
from api.engines.whisper_engine import WhisperEngine
from api.engines.diarization_engine import DiarizationEngine
from api.services.transcription_service import TranscriptionService
from api.services.diarization_service import DiarizationService
from api.services.alignment_service import AlignmentService
from api.services.operator_service import OperatorService
from api.services.embedding_service import EmbeddingService
from api.services.speaker_match_service import SpeakerMatchService
from api.repositories.operator_repository import OperatorRepository
from api.orchestrators.operator_voice_orchestrator import OperatorVoiceOrchestrator
from api.orchestrators.conversation_orchestrator import ConversationOrchestrator

_WHISPER_ENGINE = None
_DIARIZATION_ENGINE = None


def get_shared_whisper_engine() -> WhisperEngine:
    global _WHISPER_ENGINE
    if _WHISPER_ENGINE is None:
        _WHISPER_ENGINE = WhisperEngine()
        _WHISPER_ENGINE._load_model()
    return _WHISPER_ENGINE


def get_shared_diarization_engine() -> DiarizationEngine:
    global _DIARIZATION_ENGINE
    if _DIARIZATION_ENGINE is None:
        _DIARIZATION_ENGINE = DiarizationEngine()
        _DIARIZATION_ENGINE._load_pipeline()
    return _DIARIZATION_ENGINE


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_shared_diarization_engine())


def get_operator_repository() -> OperatorRepository:
    return OperatorRepository(database_session)


def get_operator_service() -> OperatorService:
    return OperatorService(get_operator_repository())


def get_alignment_service() -> AlignmentService:
    return AlignmentService()


def get_operator_voice_orchestrator() -> OperatorVoiceOrchestrator:
    return OperatorVoiceOrchestrator(get_operator_service(), get_embedding_service())


def get_conversation_orchestrator() -> ConversationOrchestrator:
    whisper_engine = get_shared_whisper_engine()
    diarization_engine = get_shared_diarization_engine()

    transcription_service = TranscriptionService(whisper_engine)
    diarization_service = DiarizationService(diarization_engine)

    operator_repo = OperatorRepository(database_session)
    operator_service = OperatorService(operator_repo)

    alignment_service = AlignmentService()

    speaker_match_service = SpeakerMatchService(operator_service)

    return ConversationOrchestrator(
        transcription_service,
        diarization_service,
        alignment_service,
        speaker_match_service,
    )
