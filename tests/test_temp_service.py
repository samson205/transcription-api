"""
Тесты для TempService

Функция get_temp_file создает временный файл на диске,
проверяя расширение файла и его размер
"""

from unittest.mock import AsyncMock
from pathlib import Path

import pytest

from api.core.config import settings
from api.services.temp_service import TempService, UnsupportedFileType, FileTooLarge

class TestTempService:
    def setup_method(self):
        self.mock_file = AsyncMock()

    @pytest.mark.parametrize(
        "filename",
        ["file.mp3", "file.wav", "file.ogg"]
    )
    async def test_valid_file(self, tmp_path, monkeypatch, filename):
        # Проходит проверку на тип, после чего начинается запись.
        # Записывается на диск и возвращает путь к файлу
        self.mock_file.filename = filename
        self.mock_file.read.side_effect = [b"fake-content", b""]
        monkeypatch.setattr(settings, "TEMP_DIR", tmp_path)

        file_path = await TempService.get_temp_file(self.mock_file)

        assert file_path.exists()
        assert file_path.parent == settings.TEMP_DIR
        assert file_path.read_bytes() == b"fake-content"
        assert file_path.suffix == Path(filename or "").suffix

    @pytest.mark.parametrize(
        "filename",
        ["file.pdf", "file.exe", "file.txt", ""]
    )
    async def test_invalid_extension(self, tmp_path, monkeypatch, filename):
        # Не проходит проверку на тип и кидает UnsupportedFileType, файл НЕ создается
        self.mock_file.filename = filename
        monkeypatch.setattr(settings, "TEMP_DIR", tmp_path)

        with pytest.raises(UnsupportedFileType) as exc_info:
            await TempService.get_temp_file(self.mock_file)

        assert self.mock_file.read.call_count == 0
        assert len([f for f in settings.TEMP_DIR.glob("**/*") if f.is_file()]) == 0
        assert "Unsupported extension" in str(exc_info.value)

    async def test_file_exceeds_size_limit(self, tmp_path, monkeypatch):
        # Проходит проверку на тип, заходит в цикл, размер превышает значение,
        # буффер закрывается, записанная часть файла удаляется с диска.
        self.mock_file.read.side_effect = [b"0" * 25, b""]
        self.mock_file.filename = "file.mp3"
        monkeypatch.setattr(settings, "TEMP_DIR", tmp_path)
        monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", 20)

        with pytest.raises(FileTooLarge) as exc_info:
            await TempService.get_temp_file(self.mock_file)

        assert len([f for f in settings.TEMP_DIR.glob("**/*") if f.is_file()]) == 0
        assert "File too large" in str(exc_info.value)
