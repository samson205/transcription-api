import torchaudio

from pyannote.core import Segment
from api.engines.embedding_engine import EmbeddingEngine


class EmbeddingService:
    def __init__(self, embedding_engine: EmbeddingEngine) -> None:
        self._embedding_engine = embedding_engine

    def load_audio(self, path: str):
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return {
            "waveform": waveform,
            "sample_rate": sample_rate
        } 

    def extract_embedding(self, audio_in_memory: dict, excerpt: Segment | None = None) -> list[float]:
        result = self._embedding_engine.extract_embedding(audio_in_memory, excerpt)
        return result.tolist() # type: ignore
