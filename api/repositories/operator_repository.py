from typing import Callable, AsyncContextManager

from sqlalchemy import select, update, delete, Row
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from api.models.operator_model import Operator
from api.models.operator_embedding_model import OperatorEmbedding
from api.models.enums import ProcessingStatus


class OperatorRepository:
    def __init__(
        self, session_factory: Callable[[], AsyncContextManager[AsyncSession]]
    ) -> None:
        self._session_factory = session_factory

    async def create(self, name: str) -> Operator:
        async with self._session_factory() as session:
            operator = Operator(name=name, status=ProcessingStatus.PENDING)
            session.add(operator)
            await session.commit()
            await session.refresh(operator)
            return operator

    async def get_by_id(self, operator_id: int) -> Operator | None:
        async with self._session_factory() as session:
            return await session.get(Operator, operator_id)

    async def get_all(self) -> list[Operator]:
        async with self._session_factory() as session:
            result = await session.scalars(select(Operator).order_by(Operator.id))
            return list(result)

    async def soft_delete(self, operator_id: int) -> bool:
        async with self._session_factory() as session:
            stmt = (
                update(Operator)
                .where(Operator.id == operator_id)
                .values(is_active=False)
                .returning(Operator.id)
            )
            result = await session.execute(stmt)
            await session.commit()

            if result.scalar_one_or_none() is None:
                return False
            return True

    async def delete(self, operator_id: int) -> None:
        async with self._session_factory() as session:
            operator = await session.get(Operator, operator_id)
            if operator is None:
                raise ValueError("Operator not found")

            await session.delete(operator)
            await session.commit()

    async def update_status(
        self, operator_id: int, status: ProcessingStatus, error_message: str | None
    ) -> None:
        async with self._session_factory() as session:
            operator = await session.get(Operator, operator_id)
            if operator is None:
                raise ValueError("Operator not found")

            operator.status = status
            operator.error_message = error_message
            await session.commit()

    async def add_embedding(self, operator_id: int, embedding: list[float], source_filename: str | None) -> OperatorEmbedding | None:
        async with self._session_factory() as session:
            operator = await session.get(Operator, operator_id)
            if operator is None:
                return None

            ref = OperatorEmbedding(
                operator_id=operator_id,
                embedding=embedding,
                source_filename=source_filename,
            )
            session.add(ref)
            operator.status = ProcessingStatus.SUCCESS
            await session.commit()
            await session.refresh(ref)
            return ref

    async def get_embeddings(self, operator_id: int) -> list[OperatorEmbedding]:
        async with self._session_factory() as session:
            result = await session.scalars(
                select(OperatorEmbedding)
                .where(OperatorEmbedding.operator_id == operator_id)
                .order_by(OperatorEmbedding.id)
            )
            return list(result.all())

    async def delete_embedding(self, embedding_id: int) -> bool:
        async with self._session_factory() as session:
            stmt = (
                delete(OperatorEmbedding)
                .where(OperatorEmbedding.id == embedding_id)
                .returning(OperatorEmbedding.id)
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.scalar_one_or_none() is not None

    async def find_nearest(
        self, embedding: list[float]
    ) -> Row[tuple[Operator, float]] | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    Operator,
                    OperatorEmbedding.embedding.cosine_distance(embedding).label("distance"),
                )
                .join(OperatorEmbedding, OperatorEmbedding.operator_id == Operator.id)
                .options(selectinload(Operator.embeddings))
                .where(
                    Operator.status == ProcessingStatus.SUCCESS,
                    Operator.is_active == True,
                )
                .order_by("distance")
                .limit(1)
            )
            return result.first()
