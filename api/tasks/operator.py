from api.core.celery import celery
from api.core.utils import run_async_coro
from api.core.dependencies import get_operator_service_for_celery


@celery.task
def extract_operator_embedding_task(operator_id: int, file_path: str):
    service = get_operator_service_for_celery()
    try:
        run_async_coro(service.update_embedding(operator_id, file_path))
        return {"status": "success", "operator_id": operator_id}
    except Exception:
        return {"status": "failed", "operator_id": operator_id}
