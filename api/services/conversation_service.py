from fastapi import UploadFile

from api.services.transcription_service import TranscriptionService
from api.services.diarization_service import DiarizationService
from api.services.alignment_service import AlignmentService
from api.services.temp_service import TempService
from api.schemas.schemas import ConversationResponse


class ConversationService:
    def __init__(self, transcription_service: TranscriptionService, diarization_service: DiarizationService, alignment_service: AlignmentService) -> None:
        self._transcription_service = transcription_service
        self._diarization_service = diarization_service
        self._alignment_service = alignment_service

    def process(self, path: str) -> ConversationResponse:
        try:
            transcription = self._transcription_service.transcribe_file(str(path))
            diarization = self._diarization_service.diarize(str(path))
            conversation = self._alignment_service.align(transcription.segments, diarization)
        finally:
            TempService.delete_temp_file(path)

        return ConversationResponse(
            language=transcription.language,
            duration=transcription.duration,
            segments=conversation
        )
    