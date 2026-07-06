from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    HF_TOKEN: str = ""
    REDIS_URL: str = ""

    MODEL_NAME: str = "medium"
    DEVICE: str = "cuda"
    COMPUTE_TYPE: str = "int8_float16"
    LOCAL_FILES_ONLY: bool = False

    MODELS_DIR: Path = BASE_DIR / "models"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
