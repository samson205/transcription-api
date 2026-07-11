import uuid
from pathlib import Path

from api.core.celery import celery
from api.core.utils import run_async_coro
from api.core.dependencies import get_conversation_orchestrator


@celery.task
def transcribe_task(task_id_str: str, path: str, original_filename: str):
    try:
        task_id = uuid.UUID(task_id_str)
        orchestrator = get_conversation_orchestrator()
        run_async_coro(
            orchestrator.process_and_get_conversation(task_id, original_filename, path)
        )
    finally:
        if Path(path).exists():
            Path(path).unlink(missing_ok=True)

    return {"status": "success", "conversation_id": str(task_id)}
