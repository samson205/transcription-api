from pathlib import Path

from api.core.celery import celery
from api.core.utils import run_async_coro
from api.core.dependencies import get_operator_voice_orchestrator


@celery.task
def extract_operator_embedding_task(operator_id: int, file_path: str):
    try:
        orchestrator = get_operator_voice_orchestrator()
        run_async_coro(orchestrator.process_and_register_voice(operator_id, file_path))
    finally:
        if Path(file_path).exists():
            Path(file_path).unlink(missing_ok=True)

    return {"status": "success", "operator_id": operator_id}
