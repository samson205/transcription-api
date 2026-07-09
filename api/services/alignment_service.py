from api.schemas.transcription import SegmentSchema, SpeakerSegment, DialogueSegment
from api.services.operator_service import OperatorService


class AlignmentService:
    def __init__(self, operator_service: OperatorService) -> None:
        self._operator_service = operator_service

    def align(self, transcription: list[SegmentSchema], diarization: list[SpeakerSegment]) -> list[DialogueSegment]:
        result = []
        for segment in transcription:
            best_speaker = None
            max_intersection = 0

            for speaker in diarization:
                intersection = min(segment.end, speaker.end) - max(segment.start, speaker.start)
                if intersection >= max_intersection:
                    max_intersection = intersection
                    best_speaker = speaker.speaker

            if not best_speaker:
                best_speaker = "Unknown speaker"
            
            result.append(
                DialogueSegment(
                    start=segment.start,
                    end=segment.end,
                    speaker=best_speaker,
                    text=segment.text
                )
            )
        return result
    
    async def match_operators(self, segments: list[DialogueSegment], embeddings: dict) -> list[DialogueSegment]:
        threshold = 0.4
        absolte_max_dist = 0.6
        speaker_mapping = {}
        scores = {}
        for speaker_id, embedding_vector in embeddings.items():
            if hasattr(embedding_vector, "tolist"):
                embedding_vector = embedding_vector.tolist()

            operator, distance = await self._operator_service.find_matching_operator(embedding_vector)
            scores[speaker_id] = {"operator": operator, "distance": distance}

        if len(scores) == 2:
            sp_1, sp_2 = list(scores.keys())
            dist_1 = scores[sp_1]["distance"]
            dist_2 = scores[sp_2]["distance"]

            if dist_1 <= threshold and dist_2 >= threshold:
                op_name = scores[sp_1]["operator"].name
                speaker_mapping[sp_1] = f"Оператор ({op_name})"
                speaker_mapping[sp_2] = "Клиент"
            elif dist_2 <= threshold and dist_1 >= threshold:
                op_name = scores[sp_2]["operator"].name
                speaker_mapping[sp_1] = "Клиент"
                speaker_mapping[sp_2] = f"Оператор ({op_name})"
            else:
                if dist_1 < dist_2 and dist_1 <= absolte_max_dist and scores[sp_1]["operator"]:
                    op_name = scores[sp_1]["operator"].name
                    speaker_mapping[sp_1] = f"Оператор ({op_name}) [Неуверенно]"
                    speaker_mapping[sp_2] = "Клиент"
                elif dist_2 < dist_1 and dist_2 <= absolte_max_dist and scores[sp_2]["operator"]:
                    op_name = scores[sp_2]["operator"].name
                    speaker_mapping[sp_1] = "Клиент"
                    speaker_mapping[sp_2] = f"Оператор ({op_name}) [Неуверенно]"
                else:
                    if dist_1 < dist_2:
                        speaker_mapping[sp_1] = "Оператор [Неизвестен / Неуверенно]"
                        speaker_mapping[sp_2] = "Клиент"
                    else:
                        speaker_mapping[sp_1] = "Клиент"
                        speaker_mapping[sp_2] = "Оператор [Неизвестен / Неуверенно]"

        else:
            for speaker_id, data in scores.items():
                if data["distance"] <= threshold and data["operator"]:
                    speaker_mapping[speaker_id] = f"Оператор ({data["operator"].name})"
                else:
                    speaker_mapping[speaker_id] = "Клиент / Неизвестно"

        matched_segments = []
        for segment in segments:
            resolved_role = speaker_mapping.get(segment.speaker, "Неизвестный")
            updated_segment = segment.model_copy(update={"speaker":resolved_role})
            matched_segments.append(updated_segment)
        return matched_segments
    