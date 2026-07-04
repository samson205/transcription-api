from fastapi import APIRouter, Depends

from api.transcription.schemas import Transcription
from api.transcription.service import TranscriptionService
from api.core.dependencies import get_transcription_service

router = APIRouter(prefix="/transcription", tags=["transcription"])


@router.post("/", response_model=Transcription)
async def transcribe(
    path: str,
    service: TranscriptionService = Depends(get_transcription_service)
):
    return service.get_transcription(path)
