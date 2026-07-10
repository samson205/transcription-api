import uuid
from typing import Callable, AsyncContextManager, Any

from sqlalchemy.ext.asyncio import AsyncSession

from api.models.conversation_model import Conversation


class ConversationRepository:
    def __init__(
        self, session_factory: Callable[[], AsyncContextManager[AsyncSession]]
    ) -> None:
        self._session_factory = session_factory

    async def create(
        self,
        task_id: uuid.UUID,
        filename: str,
        language: str,
        duration: float,
        segments: list[dict[str, Any]],
    ) -> Conversation:
        async with self._session_factory() as session:
            transcription = Conversation(
                id=task_id,
                filename=filename,
                language=language,
                duration=duration,
                segments=segments,
            )
            session.add(transcription)
            await session.commit()
            await session.refresh(transcription)
            return transcription

    async def get_by_id(self, conversation_id: uuid.UUID) -> Conversation | None:
        async with self._session_factory() as session:
            return await session.get(Conversation, conversation_id)
