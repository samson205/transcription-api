from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    TEMP_DIR: Path = BASE_DIR / "temp"
    MODELS_DIR: Path = BASE_DIR / "ml_models"

    HF_TOKEN: str = ""
    REDIS_URL: str = ""
    DB_URL: str = ""

    MODEL_NAME: str = "large-v3"
    DEVICE: str = "cuda"
    COMPUTE_TYPE: str = "int8_float16"
    LOCAL_FILES_ONLY: bool = False
    WORD_TIMESTAMPS: bool = False

    EXTENSIONS: set[str] = {".mp3", ".wav", ".ogg"}
    MAX_UPLOAD_SIZE_BYTES: int = 100 * 1024 * 1024

    THRESHOLD: float = 0.45
    UNCERTAIN_BOUND: float = 0.5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
