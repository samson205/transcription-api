from fastapi import APIRouter, UploadFile, File

from api.schemas.task import BaseTaskResponse
from api.services.task_service import TaskService
from api.services.temp_service import TempService

router = APIRouter(prefix="/transcriptions", tags=["transcriptions"])


@router.post("/", response_model=BaseTaskResponse)
async def transcribe(
    file: UploadFile = File(...),
):
    tmp_path = TempService.get_temp_file(file)
    task_id = TaskService.transcribe_task(str(tmp_path))
    return BaseTaskResponse(task_id=task_id)
