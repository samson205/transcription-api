import numpy as np

from api.engines.diarization_engine import DiarizationEngine
from api.schemas.schemas import SpeakerSegment


class DiarizationService:
    def __init__(self, engine: DiarizationEngine) -> None:
        self._engine = engine

    def diarize(self, path: str) -> tuple[list[SpeakerSegment], dict]:
        annotation = self._engine.diarize_audio(path)
        result = []
        speaker_embeddings = {}
        if not annotation:
            return result, speaker_embeddings
        
        if hasattr(annotation[0], "features"):
            for speaker in annotation[0].labels():
                speaker_segments = annotation[0].label_timeline(speaker)
                
                try:
                    features_for_speaker = annotation[0].features.crop(speaker_segments)
                    
                    if len(features_for_speaker) > 0:
                        mean_embedding = np.mean(features_for_speaker, axis=0)
                        speaker_embeddings[speaker] = mean_embedding
                except Exception as e:
                    print(f"Не удалось извлечь эмбеддинг для {speaker}: {e}")

        for turn, _, speaker in annotation[0].itertracks(yield_label=True):
            result.append(
                SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker
                )
            )
        return result, speaker_embeddings
    