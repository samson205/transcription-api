from api.core.celery import celery
from api.core.utils import run_async_coro
from api.core.dependencies import get_conversation_service_for_celery


@celery.task
def transcribe_task(path: str):
    service = get_conversation_service_for_celery()
    result = run_async_coro(service.process(path))
    return result.model_dump()

