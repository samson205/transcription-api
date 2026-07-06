from api.core.celery import celery
from api.core.dependencies import get_conversation_service


@celery.task
def transcribe_task(path: str):
    service = get_conversation_service()
    return service.process(path).model_dump()
