from uuid import uuid4
from pathlib import Path

from fastapi import UploadFile
import aiofiles

from api.core.config import settings

CHUNK_SIZE = 1024 * 1024


class UnsupportedFileType(ValueError):
    pass


class FileTooLarge(ValueError):
    pass


class TempService:
    @staticmethod
    async def get_temp_file(file: UploadFile) -> Path:
        TempService._validate_extension(file)

        Path(settings.TEMP_DIR).mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename or "").suffix
        file_path = Path(settings.TEMP_DIR) / f"{uuid4()}{suffix}"

        size = 0
        async with aiofiles.open(file_path, "wb") as buffer:
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > settings.MAX_UPLOAD_SIZE_BYTES:
                    await buffer.close()
                    file_path.unlink(missing_ok=True)
                    raise FileTooLarge(
                        f"File too large. Maximum size {settings.MAX_UPLOAD_SIZE_BYTES / 1024 / 1024} MB"
                    )
                await buffer.write(chunk)

        return file_path

    @staticmethod
    def _validate_extension(file: UploadFile) -> None:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in settings.EXTENSIONS:
            raise UnsupportedFileType(
                f"Unsupported extension '{suffix}. Allowed {settings.EXTENSIONS}'"
            )
