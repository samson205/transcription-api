import shutil
from uuid import uuid4
from pathlib import Path

from fastapi import UploadFile

from api.core.config import settings


class TempService:
    @staticmethod
    def get_temp_file(file: UploadFile) -> Path:
        Path(settings.TEMP_DIR).mkdir(parents=True, exist_ok=True)

        suffix = Path(file.filename or "").suffix
        file_path = Path(settings.TEMP_DIR) / f"{uuid4()}{suffix}"

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return file_path
    
    @staticmethod
    def delete_temp_file(path: str):
        Path(path).unlink(missing_ok=True)
    