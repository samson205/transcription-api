from api.repositories.operator_repository import OperatorRepository
from api.models.operator_model import Operator
from api.models.enums import ProcessingStatus
from api.schemas.operator import OperatorCreate


class OperatorService:
    def __init__(self, operator_repository: OperatorRepository) -> None:
        self._repo = operator_repository

    async def register(self, data: OperatorCreate) -> Operator:
        """Создает нового оператора в БД"""
        return await self._repo.create(data.name)

    async def get_by_id(self, operator_id: int) -> Operator:
        """Получает оператора по ID"""
        result = await self._repo.get_by_id(operator_id)
        if not result:
            raise ValueError("Operator not found")
        return result

    async def update_embedding(self, operator_id: int, embedding: list[float]) -> None:
        """Обновляет эмбеддинг оператора в БД"""
        result = await self._repo.update_embedding(operator_id, embedding)
        if not result:
            raise ValueError("Operator not found")
        return None

    async def update_status(
        self, operator_id: int, status: ProcessingStatus, error_message: str | None
    ):
        """Обновляет статус обработки оператора"""
        await self._repo.update_status(operator_id, status, error_message)

    async def soft_delete(self, operator_id: int) -> None:
        """Делает поле is_active = False (мягкое удаление)"""
        result = await self._repo.soft_delete(operator_id)
        if not result:
            raise ValueError("Operator not found")

    async def find_matching_operator(
        self, embedding: list[float]
    ) -> tuple[Operator | None, float]:
        """Находит лучшее совпадение оператора по эмбеддингу"""
        result = await self._repo.find_nearest(embedding)
        if result:
            matched_operator, distance = result
            return matched_operator, distance

        return None, 1.0
