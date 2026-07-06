import torch
import torchaudio
from pyannote.audio import Pipeline

from api.core.config import settings


class DiarizationEngine:
    def __init__(self) -> None:
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.HF_TOKEN
        )
        self._pipeline.to(torch.device("cuda"))

    def diarize_audio(self, path: str):
        if not self._pipeline:
            return
        waveform, sample_rate = torchaudio.load(path)
        return self._pipeline(
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
            },
            max_speakers=2
        )
