from pathlib import Path

import torch
import torchaudio
from pyannote.audio import Pipeline
from huggingface_hub import snapshot_download

from api.core.config import settings


class DiarizationEngine:
    def __init__(self) -> None:
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            pyannote_cache_dir = Path(settings.MODELS_DIR) / "pyannote"
            pyannote_cache_dir.mkdir(parents=True, exist_ok=True)

            model_dir = snapshot_download(
                repo_id="pyannote/speaker-diarization-3.1",
                repo_type="model",
                cache_dir=str(pyannote_cache_dir),
                token=settings.HF_TOKEN,
                local_files_only=settings.LOCAL_FILES_ONLY
            )
            config_path = Path(model_dir) / "config.yaml"

            self._pipeline = Pipeline.from_pretrained(
                config_path,
                use_auth_token=settings.HF_TOKEN
            )
            self._pipeline.to(torch.device(settings.DEVICE))
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
