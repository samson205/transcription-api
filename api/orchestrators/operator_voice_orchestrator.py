from api.services.operator_service import OperatorService
from api.services.embedding_service import EmbeddingService
from api.services.temp_service import TempService


class OperatorVoiceOrchestrator:
    def __init__(
        self, operator_service: OperatorService, embedding_service: EmbeddingService
    ) -> None:
        self._operator_service = operator_service
        self._embedding_service = embedding_service

    async def process_and_register_voice(
        self, operator_id: int, file_path: str
    ) -> dict:
        """Обрабатывает голос и добавляет эмбеддинг в БД"""
        try:
            embedding = self._embedding_service.extract_embedding(file_path)
            await self._operator_service.update_embedding(operator_id, embedding)
            return {"status": "success", "operator_id": operator_id}
        finally:
            TempService.delete_temp_file(file_path)
