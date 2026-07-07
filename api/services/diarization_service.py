import numpy as np

from api.engines.diarization_engine import DiarizationEngine
from api.schemas.schemas import SpeakerSegment


class DiarizationService:
    def __init__(self, engine: DiarizationEngine) -> None:
        self._engine = engine

    def diarize(self, path: str) -> tuple[list[SpeakerSegment], dict]:
        output = self._engine.diarize_audio(path)
        result = []
        speaker_embeddings = {}
        if not output:
            return result, speaker_embeddings
        
        if isinstance(output, tuple):
            annotation = output[0]
            embeddings = output[1]
        else:
            annotation = output
            embeddings = None

        if embeddings is not None and isinstance(embeddings, np.ndarray):
            labels = list(annotation.labels())

            for i, speaker in enumerate(labels):
                try:
                    speaker_embeddings[speaker] = embeddings[i]
                except Exception:
                    print("Ошибка индексации")

        for turn, _, speaker in annotation.itertracks(yield_label=True):
            result.append(
                SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker
                )
            )
        return result, speaker_embeddings
    