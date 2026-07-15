import logging

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status

from api.core.dependencies import get_conversation_service
from api.schemas.transcription import ConversationResponse, BaseConversationResponse
from api.services.temp_service import TempService, UnsupportedFileType, FileTooLarge
from api.services.conversation_service import ConversationService
from api.tasks.transcription_tasks import transcribe_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.post(
    "/", response_model=BaseConversationResponse, status_code=status.HTTP_202_ACCEPTED
)
async def transcribe(
    file: UploadFile = File(...),
    service: ConversationService = Depends(get_conversation_service),
):
    logger.info(
        "Upload recieved filename=%s content_type=%s", file.filename, file.content_type
    )
    try:
        tmp_path = await TempService.get_temp_file(file)
    except (UnsupportedFileType, FileTooLarge) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    conversation = await service.create(str(file.filename))
    task = transcribe_task.apply_async(
        args=[conversation.id, str(tmp_path), str(file.filename)]
    )
    logger.info(
        "conversation_id=%s task_id=%s Queued for transcription file=%s",
        task.id,
        conversation.id,
        file.filename,
    )
    return conversation


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: int,
    service: ConversationService = Depends(get_conversation_service),
):
    try:
        return await service.get_by_id(conversation_id)
    except ValueError as e:
        logger.warning("conversation_id=%s Not found", conversation_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
