from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    TEMP_DIR: Path = BASE_DIR / "temp"
    MODELS_DIR: Path = BASE_DIR / "ml_models"

    HF_TOKEN: str = ""
    REDIS_URL: str = ""
    DB_URL: str = ""

    MODEL_NAME: str = "medium"
    DEVICE: str = "cuda"
    COMPUTE_TYPE: str = "int8_float16"
    LOCAL_FILES_ONLY: bool = False
    WORD_TIMESTAMPS: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
