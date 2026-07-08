from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status

from api.services.operator_service import OperatorService
from api.services.temp_service import TempService
from api.core.dependencies import get_operator_service_for_api
from api.core.schemas import OperatorCreate, OperatorRead
from api.tasks.operator import extract_operator_embedding_task

router = APIRouter(prefix="/operators", tags=["operators"])


@router.post("/")
async def create_operator(
    data: OperatorCreate = Depends(OperatorCreate.as_form),
    file: UploadFile = File(...),
    service: OperatorService = Depends(get_operator_service_for_api)
):
    tmp_path = TempService.get_temp_file(file)
    operator = await service.register(data)
    task = extract_operator_embedding_task.delay(operator.id, str(tmp_path))
    return {
        "operator_id": operator.id,
        "task_id": task.id,
    }


@router.get("/{operator_id}", response_model=OperatorRead)
async def get_operator_by_id(
    operator_id: int,
    service: OperatorService = Depends(get_operator_service_for_api)
):
    try:
        result = await service.get_by_id(operator_id)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    