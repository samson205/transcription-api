from faster_whisper.transcribe import Segment, Word

from api.schemas.transcription import SegmentSchema


class SegmentAggregator:
    """Отвечает за постобработку и склейку текстовых сегментов"""

    def merge_by_sentences(self, segments: list[Segment | Word]) -> list[SegmentSchema]:
        result = []
        current = None

        for segment in segments:
            text_chunk = getattr(segment, "text", getattr(segment, "word", "")).strip()
            if current is None:
                current = SegmentSchema(
                    start=segment.start,
                    end=segment.end,
                    text=text_chunk,
                )
            else:
                current.end = segment.end
                current.text += " " + text_chunk

            if current.text.endswith((".", "!", "?")):
                result.append(current)
                current = None

        if current:
            result.append(current)

        return result
