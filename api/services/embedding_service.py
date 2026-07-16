import torch
import torchaudio
import numpy as np
from pyannote.core import Segment

from api.engines.embedding_engine import EmbeddingEngine


class EmbeddingService:
    def __init__(self, embedding_engine: EmbeddingEngine) -> None:
        self._embedding_engine = embedding_engine

    def load_audio(self, path: str) -> dict:
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            waveform = resampler(waveform)
            sample_rate = 16000

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return {"waveform": waveform, "sample_rate": sample_rate}

    def extract_embedding(
        self, audio_in_memory: dict, excerpt: Segment | None = None
    ) -> list[float]:
        result = self._embedding_engine.extract_embedding(audio_in_memory, excerpt)
        return result.tolist()  # type: ignore
    
    def extract_averaged_embedding(self, path: str, window: float = 2.0, step: float = 1.0) -> list[float]:
        audio_in_memory = self.load_audio(path)
        duration = audio_in_memory["waveform"].shape[1] / audio_in_memory["sample_rate"]

        if duration < window:
            return self.extract_embedding(audio_in_memory, excerpt=None)
        
        embeddings = []
        for excerpt in self._sliding_windows(duration, window, step):
            if self._is_silent(audio_in_memory, excerpt=excerpt):
                continue
            embeddings.append(self.extract_embedding(audio_in_memory, excerpt=excerpt))

        if not embeddings:
            raise ValueError(f"No non-silent windwos found in {path} - cannot build reference embedding")
        
        return np.mean(embeddings, axis=0).tolist()
    
    def _is_silent(self, audio_in_memory: dict, excerpt: Segment, rms_threshold: float = 0.01) -> bool:
        waveform = audio_in_memory["waveform"]
        sample_rate = audio_in_memory["sample_rate"]
        chunk = waveform[:, int(excerpt.start * sample_rate):int(excerpt.end * sample_rate)]
        if chunk.numel() == 0:
            return True
        rms = torch.sqrt(torch.mean(chunk ** 2))
        return rms.item() < rms_threshold

    def _sliding_windows(self, duration: float, window: float, step: float) -> list[Segment]:
        windows = []
        start = 0.0
        while start + window <= duration:
            windows.append(Segment(start, start + window))
            start += step

        if not windows or windows[-1].end < duration:
            tail_start = max(0.0, duration - window)
            windows.append(Segment(tail_start, duration))

        return windows
    