import logging

from pyannote.core import Segment
from scipy.spatial.distance import cosine

from api.services.operator_service import OperatorService
from api.services.embedding_service import EmbeddingService
from api.schemas.transcription import DialogueSegment

logger = logging.getLogger(__name__)


class SpeakerMatchService:
    def __init__(self, operator_service: OperatorService, embedding_service: EmbeddingService) -> None:
        self._operator_service = operator_service
        self._embedding_service = embedding_service

    async def match_operators(self, segments: list[DialogueSegment], path: str):
        # TODO: Вынести threshold из функции в конфиг
        threshold = 0.45

        audio_in_memory = self._embedding_service.load_audio(path)
        best_operator = await self._identify_operator(segments, audio_in_memory)
        if not best_operator:
            return segments
        
        target_operator_vector = best_operator.embedding
        if not target_operator_vector:
            return segments
        
        matched_segments = []
        for segment in segments:
            duration = segment.end - segment.start
            resolved_role = None
            if duration < 0.3:
                continue

            try:
                excerpt = Segment(segment.start, segment.end)
                segment_emb = self._embedding_service.extract_embedding(audio_in_memory, excerpt)

                dist_to_operator = cosine(segment_emb, target_operator_vector)
                if dist_to_operator <= threshold:
                    resolved_role = f"Оператор ({best_operator.name})"
                else:
                    resolved_role = "Клиент"
            
            except Exception as e:
                # TODO: Здесь нужен лог
                print(str(e))
                resolved_role = "Неизвестный"
            
            upd_segment = segment.model_copy(update={"speaker": resolved_role})
            matched_segments.append(upd_segment)

        return matched_segments

    async def _identify_operator(self, segments: list[DialogueSegment], audio_in_memory: dict):
        long_segments = sorted(segments, key=lambda s: (s.end - s.start), reverse=True)
        chunks_to_analyze = long_segments[:7]
        votes = {}
        operators = {}
        # TODO: Вынести threshold из функции в конфиг
        threshold = 0.45

        for segment in chunks_to_analyze:
            try:
                excerpt = Segment(segment.start, segment.end)
                segment_emb = self._embedding_service.extract_embedding(audio_in_memory, excerpt)

                operator, distance = await self._operator_service.find_matching_operator(segment_emb)

                if operator and distance <= threshold:
                    votes[operator.id] = votes.get(operator.id, 0) + 1
                    operators[operator.id] = operator

            except Exception as e:
                # TODO: Здесь нужен лог
                print(f"error: {str(e)}")

        if not votes:
            return None
        
        winner_id = max(votes, key=votes.get) # type: ignore
        winner_operator = operators[winner_id]
        return winner_operator
