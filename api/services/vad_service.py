import numpy as np

from api.engines.vad_engine import VadEngine


class VadService:
    def __init__(self, vad_engine: VadEngine) -> None:
        self._vad_engine = vad_engine

    def get_speech_regions(self, audio_in_memory: dict) -> list[tuple[float, float]]:
        waveform = audio_in_memory["waveform"]
        sample_rate = audio_in_memory["sample_rate"]
        audio_np = waveform.squeeze(0).numpy().astype(np.float32)
        return self._vad_engine.get_speech_regions(audio_np, sample_rate)

    def clip_to_speech(self, start: float, end: float, speech_regions: list[tuple[float, float]]) -> tuple[float, float] | None:
        best_overlap = 0.0
        best_region = None
        for r_start, r_end in speech_regions:
            overlap = min(end, r_end) - max(start, r_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_region = (max(start, r_start), min(end, r_end))
        return best_region
    