import logging
from pathlib import Path

from api.core.celery import celery
from api.core.utils import run_async_coro, release_gpu_memory
from api.core.dependencies import get_operator_voice_orchestrator

logger = logging.getLogger(__name__)


@celery.task
def extract_operator_embedding_task(operator_id: int, file_path: str):
    logger.info(
        "operator_id=%s Starting voice embedding extraction file=%s",
        operator_id,
        file_path,
    )
    try:
        orchestrator = get_operator_voice_orchestrator()
        run_async_coro(orchestrator.process_and_register_voice(operator_id, file_path))
        logger.info("operator_id=%s Completed", operator_id)
    except Exception:
        logger.exception(
            "operator_id=%s Failed extracting voice embedding file=%s",
            operator_id,
            file_path,
        )
        raise
    finally:
        if Path(file_path).exists():
            Path(file_path).unlink(missing_ok=True)
        release_gpu_memory()

    return {"status": "success", "operator_id": operator_id}
