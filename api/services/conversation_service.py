from typing import Any

from api.repositories.conversation_repository import ConversationRepository
from api.models.conversation_model import Conversation
from api.models.enums import ProcessingStatus
from api.schemas.transcription import DialogueSegment


class ConversationService:
    def __init__(self, repository: ConversationRepository) -> None:
        self._repository = repository

    async def create(
        self,
        filename: str,
    ) -> Conversation:
        return await self._repository.create(
            filename,
        )

    async def get_by_id(self, conversation_id: int) -> Conversation:
        result = await self._repository.get_by_id(conversation_id)
        if not result:
            raise ValueError("Conversation doesn't exist")

        return result

    async def update_status(
        self, conversation_id: int, status: ProcessingStatus, error_message: str | None
    ):
        await self._repository.update_status(conversation_id, status, error_message)

    async def save_final_result(
        self,
        conversation_id: int,
        operator_id: int | None,
        language: str,
        duration: float,
        segments: list[DialogueSegment],
    ) -> Conversation:
        return await self._repository.save_results(
            conversation_id,
            operator_id,
            language,
            duration,
            [s.model_dump() for s in segments],
        )
