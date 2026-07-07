import torch
import torchaudio
from pyannote.audio import Pipeline

from api.core.config import settings


class DiarizationEngine:
    def __init__(self) -> None:
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=settings.HF_TOKEN
            )
            self._pipeline.to(torch.device("cuda"))
        return self._pipeline

    def diarize_audio(self, path: str):
        pipeline = self._load_pipeline()
        waveform, sample_rate = torchaudio.load(path)
        return pipeline(
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
            },
            max_speakers=2,
            return_embeddings=True
        )
