import logging
import time

from api.core.config import settings
from api.core.utils import parse_call_metadata
from api.services.transcription_service import TranscriptionService
from api.services.speaker_match_service import SpeakerMatchService
from api.services.conversation_service import ConversationService
from api.services.operator_service import OperatorService
from api.services.embedding_service import EmbeddingService
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
        operator_service: OperatorService,
        embedding_service: EmbeddingService,
    ) -> None:
        self._transcription_service = transcription_service
        self._speaker_match_service = speaker_match_service
        self._segment_aggregator = segment_aggregator
        self._conversation_service = conversation_service
        self._operator_service = operator_service
        self._embedding_service = embedding_service
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
                beam_size=settings.BEAM_SIZE,
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

            start_embeddings_time = time.monotonic()
            audio_in_memory = self._embedding_service.load_audio(path)
            embeddings = self._embedding_service.extract_embeddings_for_segments(audio_in_memory, [(s.start, s.end) for s in clean_segments])
            logger.info("conversation_id=%s Calculated embeddings in %d", conversation_id, time.monotonic() - start_embeddings_time)

            metadata = parse_call_metadata(original_filename)
            operator, cluster_map = None, None
            if metadata is not None:
                operator = await self._operator_service.get_by_external_id(metadata["operator_ext"])
                if operator is not None:
                    metric.method = "claimed"
                    metric.best_operator = operator.name

            if operator is None:
                operator, cluster_map = await self._speaker_match_service.identify_operator(clean_segments, embeddings, original_filename, metric)

            conversation = self._speaker_match_service.assign_roles(clean_segments, embeddings, operator, original_filename, cluster_map)

            result = await self._conversation_service.save_final_result(
                conversation_id,
                operator.id if operator else None,
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
