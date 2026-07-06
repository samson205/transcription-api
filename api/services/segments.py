from typing import Iterable

from faster_whisper.transcribe import Segment


class SegmentsService:
    @staticmethod
    def rebuild_segments(segments: Iterable[Segment]):
        result = []
        current = None

        for segment in segments:
            if current is None:
                current = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                }
            else:
                current["end"] = segment.end
                current["text"] += " " + segment.text.strip()

            if current["text"].endswith((".", "!", "?")):
                result.append(current)
                current = None

        if current:
            result.append(current)
        
        return result
