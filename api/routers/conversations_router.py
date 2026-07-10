import uuid

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status

from api.core.dependencies import get_conversation_service
from api.schemas.task import BaseTaskResponse
from api.schemas.transcription import ConversationResponse
from api.services.task_service import TaskService
from api.services.temp_service import TempService
from api.services.conversation_service import ConversationService

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.post("/", response_model=BaseTaskResponse)
async def transcribe(
    file: UploadFile = File(...),
):
    tmp_path = TempService.get_temp_file(file)
    task_id = TaskService.create_transcribe_task(str(tmp_path), str(file.filename))
    return BaseTaskResponse(task_id=task_id)


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: uuid.UUID,
    service: ConversationService = Depends(get_conversation_service),
):
    try:
        return await service.get_by_id(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
