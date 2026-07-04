from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    MODEL_NAME: str = "medium"
    DEVICE: str = "cpu"
    COMPUTE_TYPE: str = "int8"
    LOCAL_FILES_ONLY: bool = False

    MODELS_DIR: Path = BASE_DIR / "models"


settings = Settings()
