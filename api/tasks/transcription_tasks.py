import logging
from pathlib import Path

from api.core.celery import celery
from api.core.utils import run_async_coro
from api.core.dependencies import get_conversation_orchestrator

logger = logging.getLogger(__name__)


@celery.task
def transcribe_task(conversation_id: int, path: str, original_filename: str):
    logger.info(
        "conversation_id=%s Starting transcription file=%s",
        conversation_id,
        original_filename,
    )
    try:
        orchestrator = get_conversation_orchestrator()
        run_async_coro(
            orchestrator.process_and_get_conversation(
                conversation_id, original_filename, path
            )
        )
        logger.info("conversation_id=%s Completed", conversation_id)
    except Exception:
        logger.exception(
            "conversation_id=%s Failed processing file=%s",
            conversation_id,
            original_filename,
        )
        raise
    finally:
        if Path(path).exists():
            Path(path).unlink(missing_ok=True)

    return {"status": "success", "conversation_id": conversation_id}
