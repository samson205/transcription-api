import asyncio

from api.repositories.operator_repository import OperatorRepository
from api.services.temp_service import TempService
from api.services.embedding_service import EmbeddingService
from api.models.operator import Operator
from api.core.schemas import OperatorCreate


class OperatorService:
    def __init__(self, operator_repository: OperatorRepository, embedding_service: EmbeddingService | None) -> None:
        self._repo = operator_repository
        self._embedding_service = embedding_service

    async def register(self, data: OperatorCreate) -> Operator:
        return await self._repo.create(data.name)
    
    async def get_by_id(self, operator_id: int) -> Operator:
        result = await self._repo.get_by_id(operator_id)
        if not result:
            raise ValueError("Operator not found")
        return result
    
    async def update_embedding(self, operator_id: int, file_path: str) -> Operator:
        if self._embedding_service is None:
            raise Exception
        try:
            embedding = self._embedding_service.extract_embedding(file_path)
            operator = await self._repo.update_embedding(operator_id, embedding)
            return operator
        finally:
            TempService.delete_temp_file(file_path)

    async def find_matching_operator(self, embedding: list[float]):
        result = await self._repo.find_nearest(embedding)
        if result:
            matched_operator, distance = result
            return matched_operator, distance
            
        return None, 1.0
