from typing import Callable, AsyncContextManager

from sqlalchemy import select, update, Row
from sqlalchemy.ext.asyncio import AsyncSession

from api.models.operator_model import Operator


class OperatorRepository:
    def __init__(
        self, session_factory: Callable[[], AsyncContextManager[AsyncSession]]
    ) -> None:
        self._session_factory = session_factory

    async def create(self, name: str) -> Operator:
        async with self._session_factory() as session:
            operator = Operator(name=name)
            session.add(operator)
            await session.commit()
            await session.refresh(operator)
            return operator

    async def get_by_id(self, operator_id: int) -> Operator | None:
        async with self._session_factory() as session:
            return await session.get(Operator, operator_id)

    async def get_all(self) -> list[Operator]:
        async with self._session_factory() as session:
            result = await session.scalars(select(Operator))
            return list(result)

    async def delete(self, operator_id: int) -> None:
        async with self._session_factory() as session:
            operator = await session.get(Operator, operator_id)
            if operator is None:
                raise ValueError("Operator not found")

            await session.delete(operator)
            await session.commit()

    async def update_embedding(self, operator_id: int, embedding: list[float]) -> bool:
        async with self._session_factory() as session:
            stmt = (
                update(Operator)
                .where(Operator.id == operator_id)
                .values(embedding=embedding)
                .returning(Operator.id)
            )
            result = await session.execute(stmt)
            await session.commit()

            if result.scalar_one_or_none() is None:
                return False

            return True

    async def find_nearest(
        self, embedding: list[float]
    ) -> Row[tuple[Operator, float]] | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    Operator,
                    Operator.embedding.cosine_distance(embedding).label("distance"),
                )
                .order_by("distance")
                .limit(1)
            )
            return result.first()
