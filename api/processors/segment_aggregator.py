from faster_whisper.transcribe import Segment, Word

from api.core.config import settings
from api.schemas.transcription import DialogueSegment


class SegmentAggregator:
    """Отвечает за постобработку и склейку текстовых сегментов"""

    def merge_by_sentences(
        self, segments: list[Segment | Word]
    ) -> list[DialogueSegment]:
        result = []
        current = None
        prev_end = None
        max_pause = 0.7
        max_duration = 15.0

        for segment in segments:
            text_chunk = getattr(segment, "text", getattr(segment, "word", "")).strip()
            if not text_chunk:
                continue

            if current is not None and settings.WORD_TIMESTAMPS:
                gap = (segment.start - prev_end) if prev_end is not None else 0.0
                if gap >= max_pause:
                    result.append(self._finalize_segment(current))
                    current = None

            if current is not None and settings.WORD_TIMESTAMPS:
                if segment.end - current.start > max_duration:
                    result.append(self._finalize_segment(current))
                    current = None

            if current is None:
                current = DialogueSegment(
                    start=segment.start,
                    end=segment.end,
                    text=text_chunk,
                    speaker="Неизвестный",
                )
            else:
                current.end = segment.end
                current.text += " " + text_chunk

            if current.text.endswith((".", "!", "?")):
                result.append(current)
                current = None

            prev_end = segment.end

        if current:
            result.append(self._finalize_segment(current))

        return result

    def _finalize_segment(self, segment: DialogueSegment) -> DialogueSegment:
        segment.text = segment.text.strip()
        if segment.text and not segment.text.endswith((".", "!", "?")):
            segment.text += "."
        return segment
