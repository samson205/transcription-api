import logging

from pyannote.core import Segment
from scipy.spatial.distance import cosine

from api.core.config import settings
from api.services.operator_service import OperatorService
from api.services.embedding_service import EmbeddingService
from api.schemas.transcription import DialogueSegment

logger = logging.getLogger(__name__)


class SpeakerMatchService:
    def __init__(
        self, operator_service: OperatorService, embedding_service: EmbeddingService
    ) -> None:
        self._operator_service = operator_service
        self._embedding_service = embedding_service

    async def match_operators(
        self, segments: list[DialogueSegment], path: str, original_filename: str
    ) -> tuple[list[DialogueSegment], int | None]:
        audio_in_memory = self._embedding_service.load_audio(path)

        embeddings = {}
        for segment in segments:
            excerpt = Segment(segment.start, segment.end)
            key = (segment.start, segment.end)
            try:
                embeddings[key] = self._embedding_service.extract_embedding(
                    audio_in_memory, excerpt
                )
            except Exception:
                logger.exception("Failed to extract embedding for segment")
                embeddings[key] = None

        best_operator = await self._identify_operator(
            segments, embeddings, original_filename
        )
        if not best_operator:
            return segments, None

        target_operator_vector = best_operator.embedding
        if not target_operator_vector:
            return segments, None

        matched_segments = []
        for segment in segments:
            duration = segment.end - segment.start
            resolved_role = None
            if duration < 0.3:
                resolved_role = "Неизвестный"
            else:
                key = (segment.start, segment.end)
                segment_emb = embeddings[key]
                if segment_emb is None:
                    resolved_role = "Неизвестный"
                    continue

                dist_to_operator = cosine(segment_emb, target_operator_vector)
                if dist_to_operator <= settings.THRESHOLD:
                    resolved_role = f"Оператор ({best_operator.name})"
                elif dist_to_operator <= settings.UNCERTAIN_BOUND:
                    resolved_role = f"Оператор ({best_operator.name}) [Неуверенно]"
                else:
                    resolved_role = "Клиент"

            upd_segment = segment.model_copy(update={"speaker": resolved_role})
            matched_segments.append(upd_segment)

        return matched_segments, best_operator.id

    async def _identify_operator(
        self, segments: list[DialogueSegment], embeddings: dict, original_filename: str
    ):
        candidates = [s for s in segments if 2 <= s.end - s.start <= 10]
        step = max(1, len(candidates) // 20)
        chunks_to_analyze = candidates[::step][:20]

        votes = {}
        operators = {}

        for segment in chunks_to_analyze:
            key = (segment.start, segment.end)
            segment_emb = embeddings[key]
            segment_duration = segment.end - segment.start

            operator, distance = await self._operator_service.find_matching_operator(
                segment_emb
            )
            if not operator:
                continue

            if distance <= settings.UNCERTAIN_BOUND:
                weight = max(0.0, 1.0 - distance / settings.UNCERTAIN_BOUND) * min(
                    segment_duration / 3.0, 1.0
                )
                votes[operator.id] = votes.get(operator.id, 0) + weight
                operators[operator.id] = operator

        if not votes:
            logger.warning("No suitable operator found file=%s", original_filename)
            return None

        winner_id = max(votes, key=votes.get)  # type: ignore
        if votes[winner_id] < 0.5:
            logger.warning("No suitable operator found file=%s", original_filename)
            return None
        return operators[winner_id]
