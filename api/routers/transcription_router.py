from fastapi import APIRouter, UploadFile, File, Depends

from api.schemas.schemas import Transcription
from api.services.task import TaskService
from api.services.temp_storage import TempStorage

router = APIRouter(prefix="/transcriptions", tags=["transcriptions"])


@router.post("/")
async def transcribe(
    file: UploadFile = File(...),
):
    tmp_path = TempStorage.get_temp_file(file)
    return {"task_id": TaskService.create(str(tmp_path))}
