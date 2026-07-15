from typing import Callable, AsyncContextManager, Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from api.models.conversation_model import Conversation
from api.models.enums import ProcessingStatus


class ConversationRepository:
    def __init__(
        self, session_factory: Callable[[], AsyncContextManager[AsyncSession]]
    ) -> None:
        self._session_factory = session_factory

    async def create(
        self,
        filename: str,
    ) -> Conversation:
        async with self._session_factory() as session:
            conversation = Conversation(
                filename=filename,
                status=ProcessingStatus.PENDING,
            )
            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)
            return conversation

    async def get_by_id(self, conversation_id: int) -> Conversation | None:
        async with self._session_factory() as session:
            return await session.get(Conversation, conversation_id)

    async def update_status(
        self, conversation_id: int, status: ProcessingStatus, error_message: str | None
    ) -> None:
        async with self._session_factory() as session:
            conversation = await session.get(Conversation, conversation_id)
            if conversation is None:
                raise ValueError("Conversation not found")

            conversation.status = status
            conversation.error_message = error_message
            await session.commit()

    async def save_results(
        self,
        conversation_id: int,
        operator_id: int | None,
        language: str,
        duration: float,
        segments: list[dict[str, Any]],
    ):
        async with self._session_factory() as session:
            stmt = (
                update(Conversation)
                .where(Conversation.id == conversation_id)
                .values(
                    operator_id=operator_id,
                    language=language,
                    duration=duration,
                    segments=segments,
                    status=ProcessingStatus.SUCCESS,
                )
                .returning(Conversation)
            )
            result = await session.execute(stmt)
            conversation = result.scalar_one_or_none()
            if conversation is None:
                raise ValueError("Conversation not found")

            await session.commit()
            return conversation
