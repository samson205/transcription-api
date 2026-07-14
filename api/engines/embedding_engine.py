from pathlib import Path

import torch
from pyannote.core import Segment
from pyannote.audio import Model, Inference
from huggingface_hub import snapshot_download

from api.core.config import settings


class EmbeddingEngine:
    def __init__(self) -> None:
        self._inference = None

    def _load_inference(self):
        if self._inference is None:
            pyannote_cache_dir = Path(settings.MODELS_DIR) / "pyannote"
            pyannote_cache_dir.mkdir(parents=True, exist_ok=True)

            model_dir = snapshot_download(
                repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
                repo_type="model",
                cache_dir=str(pyannote_cache_dir),
                token=settings.HF_TOKEN,
                local_files_only=settings.LOCAL_FILES_ONLY,
            )
            checkpoint_path = Path(model_dir) / "pytorch_model.bin"

            model = Model.from_pretrained(
                checkpoint_path, use_auth_token=settings.HF_TOKEN
            )
            model.to(torch.device(settings.DEVICE))

            self._inference = Inference(
                model, window="whole"
            )
        return self._inference
    
    def extract_embedding(self, audio_in_memory: dict, excerpt: Segment | None):
        inference = self._load_inference()
        if excerpt is None:
            return inference(audio_in_memory)
        return inference.crop(audio_in_memory, excerpt)
