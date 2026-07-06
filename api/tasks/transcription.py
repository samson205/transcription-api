from api.core.celery import celery
from api.core.dependencies import _transcription_service


@celery.task
def transcribe_task(path: str):
    result = _transcription_service.transcribe_file(path)
    return result.model_dump()
