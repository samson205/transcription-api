import uuid

from api.repositories.conversation_repository import ConversationRepository
from api.models.conversation_model import Conversation
from api.schemas.transcription import DialogueSegment


class ConversationService:
    def __init__(self, repository: ConversationRepository) -> None:
        self._repository = repository

    async def create(
        self,
        task_id: uuid.UUID,
        filename: str,
        language: str,
        duration: float,
        segments: list[DialogueSegment],
    ) -> Conversation:
        return await self._repository.create(
            task_id, filename, language, duration, [s.model_dump() for s in segments]
        )

    async def get_by_id(self, conversation_id: uuid.UUID) -> Conversation:
        result = await self._repository.get_by_id(conversation_id)
        if not result:
            raise ValueError("Conversation not ready or doesn't exists")
        return result
