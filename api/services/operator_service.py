from typing import AsyncContextManager, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.models.operator import Operator


class OperatorService:
    def __init__(self, session_factory: Callable[[], AsyncContextManager[AsyncSession]]) -> None:
        self._session_factory = session_factory

    async def create_operator(self,):
        pass

    async def find_matching_operator(self, embedding: list[float]):
        async with self._session_factory() as session:
            result = await session.scalars(
                select(Operator, Operator.embedding.cosine_distance(embedding).label("distance"))
                .order_by("distance")
                .limit(1)
            )

        result = result.first()
        if result:
            matched_operator, distance = result
            return matched_operator, distance
            
        return None, 1.0
    