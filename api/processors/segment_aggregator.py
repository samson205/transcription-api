from typing import Iterable

from faster_whisper.transcribe import Segment

from api.schemas.transcription import SegmentSchema


class SegmentAggregator:
    """Отвечает за постобработку и склейку текстовых сегментов"""

    def merge_by_sentences(self, segments: Iterable[Segment]) -> list[SegmentSchema]:
        result = []
        current = None

        for segment in segments:
            if current is None:
                current = SegmentSchema.model_validate(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                    }
                )
            else:
                current.end = segment.end
                current.text += " " + segment.text.strip()

            if current.text.endswith((".", "!", "?")):
                result.append(current)
                current = None

        if current:
            result.append(current)

        return result
