import logging
import time

from api.core.config import settings
from api.services.transcription_service import TranscriptionService
from api.services.speaker_match_service import SpeakerMatchService
from api.services.conversation_service import ConversationService
from api.services.metrics_service import FileMetricsService
from api.processors.segment_aggregator import SegmentAggregator
from api.schemas.transcription import ConversationResponse
from api.schemas.metrics import FileMetric
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
        self._metrics_service = FileMetricsService("export/file_metrics.csv")

    async def process_and_get_conversation(
        self, conversation_id: int, original_filename: str, path: str
    ) -> ConversationResponse:
        """Обрабатывает аудиофайл и записывает транскрипцию разговора в БД"""
        logger.info(
            "conversation_id=%s Conversation pipeline started file=%s",
            conversation_id,
            original_filename,
        )

        try:
            metric = FileMetric(
                filename=original_filename,
                device=settings.DEVICE,
                compute_type=settings.COMPUTE_TYPE,
                whisper_model=settings.MODEL_NAME,
            )
            start = time.monotonic()
            await self._conversation_service.update_status(
                conversation_id, ProcessingStatus.PROCESSING, None
            )

            transcription = self._transcription_service.transcribe_file(
                str(path), original_filename
            )
            metric.file_duration = transcription.duration

            clean_segments = self._segment_aggregator.merge_by_sentences(
                transcription.segments
            )
            logger.info(
                "conversation_id=%s Aggregated into %d sentences",
                conversation_id,
                len(clean_segments),
            )

            conversation, operator_id = (
                await self._speaker_match_service.match_operators(
                    clean_segments, path, original_filename, metric
                )
            )

            result = await self._conversation_service.save_final_result(
                conversation_id,
                operator_id,
                transcription.language,
                transcription.duration,
                conversation,
            )
            took_seconds = time.monotonic() - start
            metric.processing_time = round(took_seconds, 2)
            logger.info(
                "conversation_id=%s Conversation pipeline finished, conversation saved took=%ds",
                conversation_id,
                took_seconds,
            )
            if settings.DEBUG_METRICS:
                self._metrics_service.append(metric)

        except Exception as e:
            result = await self._conversation_service.update_status(
                conversation_id, ProcessingStatus.FAILURE, str(e)
            )
            raise e

        return ConversationResponse.model_validate(result)
