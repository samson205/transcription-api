from api.services.transcription_service import TranscriptionService
from api.services.diarization_service import DiarizationService
from api.services.alignment_service import AlignmentService
from api.services.temp_service import TempService
from api.services.speaker_match_service import SpeakerMatchService
from api.schemas.transcription import ConversationResponse


class ConversationOrchestrator:
    def __init__(
        self,
        transcription_service: TranscriptionService,
        diarization_service: DiarizationService,
        alignment_service: AlignmentService,
        speaker_match_service: SpeakerMatchService,
    ) -> None:
        self._transcription_service = transcription_service
        self._diarization_service = diarization_service
        self._alignment_service = alignment_service
        self._speaker_match_service = speaker_match_service

    async def process_and_get_conversation(self, path: str) -> ConversationResponse:
        """Обрабатывает аудиофайл и возвращает транскрипция разговора"""
        try:
            transcription = self._transcription_service.transcribe_file(str(path))
            diarization, embeddings = self._diarization_service.diarize(str(path))
            conversation_raw = self._alignment_service.align(
                transcription.segments, diarization
            )
            conversation = await self._speaker_match_service.match_operators(
                conversation_raw, embeddings
            )
        finally:
            TempService.delete_temp_file(path)

        return ConversationResponse(
            language=transcription.language,
            duration=transcription.duration,
            segments=conversation,
        )
