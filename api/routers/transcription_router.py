from fastapi import APIRouter, UploadFile, File

from api.core.schemas import BaseTaskResponse
from api.services.task_service import TaskService
from api.services.temp_service import TempService

router = APIRouter(prefix="/transcriptions", tags=["transcriptions"])


@router.post("/", response_model=BaseTaskResponse)
async def transcribe(
    file: UploadFile = File(...),
):
    tmp_path = TempService.get_temp_file(file)
    return BaseTaskResponse(task_id=TaskService.create(str(tmp_path)))
