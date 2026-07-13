from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status

from api.services.operator_service import OperatorService
from api.services.temp_service import TempService, UnsupportedFileType, FileTooLarge
from api.services.task_service import TaskService
from api.core.dependencies import get_operator_service
from api.schemas.operator import OperatorCreate, OperatorRead

router = APIRouter(prefix="/operators", tags=["Operators"])


@router.post("/")
async def create_operator(
    data: OperatorCreate = Depends(OperatorCreate.as_form),
    file: UploadFile = File(...),
    service: OperatorService = Depends(get_operator_service),
):
    try:
        tmp_path = await TempService.get_temp_file(file)
    except (UnsupportedFileType, FileTooLarge) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    operator = await service.register(data)
    task_id = TaskService.create_extract_operator_embedding_task(
        operator.id, str(tmp_path)
    )
    return {
        "operator_id": operator.id,
        "task_id": task_id,
    }


@router.get("/{operator_id}", response_model=OperatorRead)
async def get_operator_by_id(
    operator_id: int, service: OperatorService = Depends(get_operator_service)
):
    try:
        return await service.get_by_id(operator_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
