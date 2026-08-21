import logging

import torch
import torchaudio
import numpy as np
from pyannote.core import Segment

from api.engines.embedding_engine import EmbeddingEngine

logger = logging.getLogger(__name__)


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

        waveform = self._compress_dynamic_range(waveform)

        return {"waveform": waveform, "sample_rate": sample_rate}

    def extract_embedding(
        self, audio_in_memory: dict, excerpt: Segment | None = None
    ) -> list[float]:
        result = self._embedding_engine.extract_embedding(audio_in_memory, excerpt)
        return result.tolist()  # type: ignore

    def extract_averaged_embedding(
        self, path: str, window: float = 2.0, step: float = 1.0
    ) -> list[float]:
        audio_in_memory = self.load_audio(path)
        duration = audio_in_memory["waveform"].shape[1] / audio_in_memory["sample_rate"]

        embeddings = self._extract_window_embeddings(audio_in_memory, 0, duration, window, step)

        if not embeddings:
            raise ValueError(
                f"No non-silent windows found in {path} - cannot build reference embedding"
            )

        averaged = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(averaged)
        if norm > 1e-8:
            averaged /= norm

        return averaged.tolist()

    def extract_embeddings_for_segments(
        self, audio_in_memory: dict, segments: list[tuple[float, float]], averaged: bool = False, window: float = 2.0, step: float = 1.0
    ) -> dict[tuple[float, float], list[float] | None]:
        embeddings: dict[tuple[float, float], list[float] | None] = {}
        for start, end in segments:
            key = (start, end)

            try:
                if averaged:
                    embeddings[key] = self.extract_averaged_segment_embedding(
                        audio_in_memory, start, end, window, step,
                    )
                else:
                    embeddings[key] = self.extract_embedding(
                        audio_in_memory, Segment(start, end)
                    )

            except ValueError as e:
                logger.warning(str(e))
                embeddings[key] = None

            except Exception:
                logger.exception("Failed to extract embedding for segment %s", key)
                embeddings[key] = None

        return embeddings

    def extract_averaged_segment_embedding(self, audio_in_memory: dict, start: float, end: float, window: float = 2.0, step: float = 1.0) -> list[float]:
        embeddings = self._extract_window_embeddings(audio_in_memory, start, end, window, step)
        if not embeddings:
            raise ValueError(f"No non-silent windows found in segment {start}-{end}")

        averaged = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(averaged)
        if norm > 1e-8:
            averaged /= norm

        return averaged.tolist()

    def _extract_window_embeddings(self, audio_in_memory: dict, start: float, end: float, window: float = 2.0, step: float = 1.0) -> list[list[float]]:
        embeddings = []
        duration = end - start

        if duration <= 0:
            raise ValueError(f"Invalid segment: {start}-{end}")

        if duration <= window:
            if self._is_silent(audio_in_memory, Segment(start, end)):
                return embeddings

            return [self.extract_embedding(audio_in_memory, Segment(start, end))]

        for relative_window in self._sliding_windows(duration, window, step):
            excerpt = Segment(start + relative_window.start, start + relative_window.end)

            if self._is_silent(audio_in_memory, excerpt):
                continue

            embeddings.append(self.extract_embedding(audio_in_memory, excerpt))

        return embeddings

    def _compress_dynamic_range(
        self, waveform: torch.Tensor, threshold: float = 0.1, ratio: float = 4.0
    ) -> torch.Tensor:
        abs_waveform = waveform.abs()
        over_threshold = abs_waveform > threshold

        compressed = waveform.clone()
        compressed[over_threshold] = torch.sign(waveform[over_threshold]) * (
            threshold + (abs_waveform[over_threshold] - threshold) / ratio
        )
        return compressed

    def _is_silent(
        self, audio_in_memory: dict, excerpt: Segment, rms_threshold: float = 0.01
    ) -> bool:
        waveform = audio_in_memory["waveform"]
        sample_rate = audio_in_memory["sample_rate"]
        chunk = waveform[
            :, int(excerpt.start * sample_rate) : int(excerpt.end * sample_rate)
        ]
        if chunk.numel() == 0:
            return True
        rms = torch.sqrt(torch.mean(chunk**2))
        return rms.item() < rms_threshold

    def _sliding_windows(
        self, duration: float, window: float, step: float
    ) -> list[Segment]:
        windows = []
        start = 0.0
        while start + window <= duration:
            windows.append(Segment(start, start + window))
            start += step

        if not windows or windows[-1].end < duration:
            tail_start = max(0.0, duration - window)
            windows.append(Segment(tail_start, duration))

        return windows
