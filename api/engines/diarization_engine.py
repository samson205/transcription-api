import logging
import time
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F
from pyannote.audio import Pipeline
from huggingface_hub import snapshot_download

from api.core.config import settings

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger(__name__)


class DiarizationEngine:
    def __init__(self) -> None:
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            logger.info(
                "Loading pyannote diarizationp pipeline device=%s", settings.DEVICE
            )
            start = time.monotonic()
            pyannote_cache_dir = Path(settings.MODELS_DIR) / "pyannote"
            pyannote_cache_dir.mkdir(parents=True, exist_ok=True)

            model_dir = snapshot_download(
                repo_id="pyannote/speaker-diarization-3.1",
                repo_type="model",
                cache_dir=str(pyannote_cache_dir),
                token=settings.HF_TOKEN,
                local_files_only=settings.LOCAL_FILES_ONLY,
            )
            config_path = Path(model_dir) / "config.yaml"

            self._pipeline = Pipeline.from_pretrained(
                config_path, use_auth_token=settings.HF_TOKEN
            )
            self._pipeline.to(torch.device(settings.DEVICE))

            HYPER_PARAMETERS = {
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": 9,
                    "threshold": 0.45,
                }
            }

            self._pipeline.instantiate(HYPER_PARAMETERS)

            logger.info(
                "Diarization pipeline loaded in %.2fs", time.monotonic() - start
            )
        return self._pipeline

    def diarize_audio(self, path: str):
        pipeline = self._load_pipeline()
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != 16000:
            waveform = F.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        logger.info("Diarizating file=%s", path)
        start = time.monotonic()
        result = pipeline(
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
            },
            min_speakers=2,
            max_speakers=2,
            return_embeddings=True,
        )
        logger.info("Diarized file=%s in %.2fs", path, time.monotonic() - start)
        return result
