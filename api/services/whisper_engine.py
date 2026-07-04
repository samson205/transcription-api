from faster_whisper import WhisperModel

from api.core.config import settings


class WhisperEngine:
    def __init__(self) -> None:
        self._model = WhisperModel(
            settings.MODEL_NAME,
            device=settings.DEVICE,
            compute_type=settings.COMPUTE_TYPE,
            download_root=str(settings.MODELS_DIR),
            local_files_only=True
        )

    def transcribe(self, path: str):
        return self._model.transcribe(
            path,
            beam_size=7,
            vad_filter=True,
            condition_on_previous_text=False,
        )
    