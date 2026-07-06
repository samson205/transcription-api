from api.engines.diarization_engine import DiarizationEngine
from api.schemas.schemas import SpeakerSegment


class DiarizationService:
    def __init__(self, engine: DiarizationEngine) -> None:
        self._engine = engine

    def diarize(self, path: str) -> list[SpeakerSegment]:
        annotation = self._engine.diarize_audio(path)
        result = []
        if not annotation:
            return result

        for turn, _, speaker in annotation.itertracks(yield_label=True):
            result.append(
                SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker
                )
            )
        return result
    