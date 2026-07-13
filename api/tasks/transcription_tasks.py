import logging
import uuid
from pathlib import Path

from api.core.celery import celery
from api.core.utils import run_async_coro
from api.core.dependencies import get_conversation_orchestrator

logger = logging.getLogger(__name__)


@celery.task
def transcribe_task(task_id_str: str, path: str, original_filename: str):
    logger.info("task_id=%s Starting transcription file=%s", task_id_str, original_filename)
    try:
        task_id = uuid.UUID(task_id_str)
        orchestrator = get_conversation_orchestrator()
        run_async_coro(
            orchestrator.process_and_get_conversation(task_id, original_filename, path)
        )
        logger.info("task_id=%s Completed", task_id_str)
    except Exception:
        logger.exception("task_id=%s Failed processing file=%s", task_id_str, original_filename)
        raise
    finally:
        if Path(path).exists():
            Path(path).unlink(missing_ok=True)

    return {"status": "success", "conversation_id": str(task_id)}
