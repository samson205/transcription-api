import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import UploadFile


class TempService:
    @staticmethod
    def get_temp_file(file: UploadFile) -> Path:
        suffix = Path(str(file.filename)).suffix
        with NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        return Path(tmp_path)
    
    @staticmethod
    def delete_temp_file(path: str):
        Path(path).unlink()
    