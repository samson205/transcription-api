import logging
import time

from api.services.transcription_service import TranscriptionService
from api.services.speaker_match_service import SpeakerMatchService
from api.services.conversation_service import ConversationService
from api.processors.segment_aggregator import SegmentAggregator
from api.schemas.transcription import ConversationResponse
from api.models.enums import ProcessingStatus

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    def __init__(
        self,
        transcription_service: TranscriptionService,
        speaker_match_service: SpeakerMatchService,
        segment_aggregator: SegmentAggregator,
        conversation_service: ConversationService,
    ) -> None:
        self._transcription_service = transcription_service
        self._speaker_match_service = speaker_match_service
        self._segment_aggregator = segment_aggregator
        self._conversation_service = conversation_service

    async def process_and_get_conversation(
        self, conversation_id: int, original_filename: str, path: str
    ) -> ConversationResponse:
        """Обрабатывает аудиофайл и записывает транскрипцию разговора в БД"""
        logger.info(
            "conversation_id=%s Pipeline started file=%s",
            conversation_id,
            original_filename,
        )

        try:
            start = time.monotonic()
            await self._conversation_service.update_status(
                conversation_id, ProcessingStatus.PROCESSING, None
            )

            transcription = self._transcription_service.transcribe_file(str(path))

            clean_segments = self._segment_aggregator.merge_by_sentences(
                transcription.segments
            )
            logger.info(
                "conversation_id=%s Aggregated into %d sentences",
                conversation_id,
                len(clean_segments),
            )

            conversation, operator_id = (
                await self._speaker_match_service.match_operators(clean_segments, path)
            )

            result = await self._conversation_service.save_final_result(
                conversation_id,
                operator_id,
                transcription.language,
                transcription.duration,
                conversation,
            )
            took_seconds = time.monotonic() - start
            logger.info(
                "conversation_id=%s Pipeline finished, conversation saved took=%ds",
                conversation_id,
                took_seconds,
            )

        except Exception as e:
            result = await self._conversation_service.update_status(
                conversation_id, ProcessingStatus.FAILURE, str(e)
            )
            raise

        return ConversationResponse.model_validate(result)
