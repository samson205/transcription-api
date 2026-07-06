from api.core.celery import celery
from api.services.transcription import TranscriptionService


@celery.task
def transcribe_task(path: str):
    service = TranscriptionService()
    return service.transcribe_file(path).model_dump()
