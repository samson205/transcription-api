from api.core.database import database_session
from api.engines.whisper_engine import WhisperEngine
from api.engines.diarization_engine import DiarizationEngine
from api.services.transcription_service import TranscriptionService
from api.services.diarization_service import DiarizationService
from api.services.alignment_service import AlignmentService
from api.services.conversation_service import ConversationService
from api.services.operator_service import OperatorService

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


def get_operator_service() -> OperatorService:
    return OperatorService(database_session)


def get_alignment_service() -> AlignmentService:
    return AlignmentService(get_operator_service())


def get_conversation_service() -> ConversationService:
    return ConversationService(
        TranscriptionService(get_shared_whisper_engine()),
        DiarizationService(get_shared_diarization_engine()),
        get_alignment_service(),
    )
