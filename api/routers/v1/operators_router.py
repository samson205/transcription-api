import logging

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status

from api.services.operator_service import OperatorService
from api.services.temp_service import TempService, UnsupportedFileType, FileTooLarge
from api.core.dependencies import get_operator_service
from api.schemas.operator import OperatorCreate, OperatorRead, OperatorUpdate
from api.tasks.operator_tasks import extract_operator_embedding_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/operators", tags=["Operators"])


@router.post("/", response_model=OperatorRead, status_code=status.HTTP_202_ACCEPTED)
async def create_operator(
    data: OperatorCreate = Depends(OperatorCreate.as_form),
    file: UploadFile = File(...),
    service: OperatorService = Depends(get_operator_service),
):
    logger.info(
        "Operator registration requested name=%s filename=%s content_type=%s",
        data.name,
        file.filename,
        file.content_type,
    )
    try:
        tmp_path = await TempService.get_temp_file(file)
    except (UnsupportedFileType, FileTooLarge) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    operator = await service.register(data)
    task = extract_operator_embedding_task.apply_async(
        args=[operator.id, str(tmp_path), file.filename]
    )
    logger.info(
        "operator_id=%s task_id=%s Queued for voice embedding extraction",
        operator.id,
        task.id,
    )
    return operator


@router.get("/", response_model=list[OperatorRead])
async def get_all_operators(service: OperatorService = Depends(get_operator_service)):
    return await service.get_all()


@router.get("/{operator_id}", response_model=OperatorRead)
async def get_operator_by_id(
    operator_id: int, service: OperatorService = Depends(get_operator_service)
):
    try:
        return await service.get_by_id(operator_id)
    except ValueError as e:
        logger.warning("operator_id=%s Not found", operator_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.patch("/{operator_id}", response_model=OperatorRead)
async def update_operator_info(
    operator_id: int,
    data: OperatorUpdate,
    service: OperatorService = Depends(get_operator_service),
):
    try:
        return await service.update_info(operator_id, data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/{operator_id}", status_code=status.HTTP_204_NO_CONTENT)
async def soft_delete(
    operator_id: int, service: OperatorService = Depends(get_operator_service)
):
    try:
        await service.soft_delete(operator_id)
    except ValueError as e:
        logger.warning("operator_id=%s Not found", operator_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/{operator_id}/embeddings",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=OperatorRead,
)
async def add_operator_embedding(
    operator_id: int,
    file: UploadFile = File(...),
    service: OperatorService = Depends(get_operator_service),
):
    try:
        operator = await service.get_by_id(operator_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    try:
        tmp_path = await TempService.get_temp_file(file)
    except (UnsupportedFileType, FileTooLarge) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    task = extract_operator_embedding_task.apply_async(
        args=[operator_id, str(tmp_path), file.filename]
    )
    logger.info(
        "operator_id=%s task_id=%s Queued for additional embedding extraction",
        operator_id,
        task.id,
    )
    return operator
