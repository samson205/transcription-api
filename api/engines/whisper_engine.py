from faster_whisper import WhisperModel

from api.core.config import settings


class WhisperEngine:
    def __init__(self) -> None:
        self._model = None

    def _load_model(self):
        if self._model is None:
            models_dir = settings.MODELS_DIR / "whisper"
            models_dir.mkdir(exist_ok=True, parents=True)
            self._model = WhisperModel(
                settings.MODEL_NAME,
                device=settings.DEVICE,
                compute_type=settings.COMPUTE_TYPE,
                download_root=str(models_dir),
                local_files_only=settings.LOCAL_FILES_ONLY
            )
        return self._model

    def transcribe(self, path: str):
        model = self._load_model()
        return model.transcribe(
            path,
            beam_size=7,
            vad_filter=True,
            condition_on_previous_text=True,
        )
    