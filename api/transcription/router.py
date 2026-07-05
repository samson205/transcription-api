from fastapi import APIRouter, UploadFile, File, Depends

from api.transcription.schemas import Transcription
from api.transcription.service import TranscriptionService
from api.core.dependencies import get_transcription_service

router = APIRouter(prefix="/transcriptions", tags=["transcriptions"])


@router.post("/", response_model=Transcription)
async def transcribe(
    file: UploadFile = File(...),
    service: TranscriptionService = Depends(get_transcription_service)
):
    return service.transcribe_file(file)
