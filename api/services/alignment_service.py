from api.schemas.transcription import SegmentSchema, SpeakerSegment, DialogueSegment


class AlignmentService:
    def align(
        self, transcription: list[SegmentSchema], diarization: list[SpeakerSegment]
    ) -> list[DialogueSegment]:
        """Сопоставляет тайминги транскрипции и диаризации"""
        result = []
        for segment in transcription:
            best_speaker = None
            max_intersection = 0

            for speaker in diarization:
                intersection = min(segment.end, speaker.end) - max(
                    segment.start, speaker.start
                )
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
                    text=segment.text,
                )
            )
        return result
