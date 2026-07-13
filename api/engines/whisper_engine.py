import logging
import time

from faster_whisper import WhisperModel

from api.core.config import settings

logger = logging.getLogger(__name__)


class WhisperEngine:
    def __init__(self, word_timestamps: bool = True) -> None:
        self._model = None
        self._word_timestamps = word_timestamps

    def _load_model(self):
        if self._model is None:
            logger.info(
                "Loading Whisper model=%s, device=%s, compute_type=%s",
                settings.MODEL_NAME,
                settings.DEVICE,
                settings.COMPUTE_TYPE,
            )
            start = time.monotonic()
            models_dir = settings.MODELS_DIR / "whisper"
            models_dir.mkdir(exist_ok=True, parents=True)
            self._model = WhisperModel(
                settings.MODEL_NAME,
                device=settings.DEVICE,
                compute_type=settings.COMPUTE_TYPE,
                download_root=str(models_dir),
                local_files_only=settings.LOCAL_FILES_ONLY,
            )
            logger.info("Whisper modle loaded in %.2fs", time.monotonic() - start)
        return self._model

    def transcribe(self, path: str):
        model = self._load_model()
        return model.transcribe(
            path,
            beam_size=7,
            vad_filter=True,
            condition_on_previous_text=True,
            word_timestamps=self._word_timestamps,
        )
