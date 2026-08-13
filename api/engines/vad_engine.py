import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps


class VadEngine:
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 100,
    ) -> None:
        self._vad_options = VadOptions(
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

    def get_speech_regions(self, audio: np.ndarray, sampling_rate: int = 16000) -> list[tuple[float, float]]:
        chunks = get_speech_timestamps(
            audio, sampling_rate=sampling_rate
        )
        return [
            (chunk["start"] / sampling_rate, chunk["end"] / sampling_rate)
            for chunk in chunks
        ]
    