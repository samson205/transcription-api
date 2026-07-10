import uuid

from api.services.transcription_service import TranscriptionService
from api.services.diarization_service import DiarizationService
from api.services.alignment_service import AlignmentService
from api.services.temp_service import TempService
from api.services.speaker_match_service import SpeakerMatchService
from api.services.conversation_service import ConversationService
from api.processors.segment_aggregator import SegmentAggregator


class ConversationOrchestrator:
    def __init__(
        self,
        transcription_service: TranscriptionService,
        diarization_service: DiarizationService,
        alignment_service: AlignmentService,
        speaker_match_service: SpeakerMatchService,
        segment_aggregator: SegmentAggregator,
        conversation_service: ConversationService,
    ) -> None:
        self._transcription_service = transcription_service
        self._diarization_service = diarization_service
        self._alignment_service = alignment_service
        self._speaker_match_service = speaker_match_service
        self._segment_aggregator = segment_aggregator
        self._conversation_service = conversation_service

    async def process_and_get_conversation(
        self, task_id: uuid.UUID, original_filename: str, path: str
    ) -> None:
        """Обрабатывает аудиофайл и записывает транскрипцию разговора в БД"""
        try:
            transcription = self._transcription_service.transcribe_file(str(path))
            clean_segments = self._segment_aggregator.merge_by_sentences(
                transcription.segments
            )
            diarization, embeddings = self._diarization_service.diarize(str(path))
            conversation_raw = self._alignment_service.align(
                clean_segments, diarization
            )
            conversation = await self._speaker_match_service.match_operators(
                conversation_raw, embeddings
            )
            await self._conversation_service.create(
                task_id,
                original_filename,
                transcription.language,
                transcription.duration,
                conversation,
            )
        finally:
            TempService.delete_temp_file(path)
