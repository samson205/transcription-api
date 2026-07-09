from api.core.celery import celery
from api.core.utils import run_async_coro
from api.core.dependencies import get_conversation_orchestrator


@celery.task
def transcribe_task(path: str):
    orchestrator = get_conversation_orchestrator()
    result = run_async_coro(orchestrator.process_and_get_conversation(path))
    return result.model_dump()
