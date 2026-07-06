from fastapi import APIRouter

from api.schemas.schemas import Transcription
from api.services.task import TaskService

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}")
async def get_task(task_id: str):
    task = TaskService.status(task_id)

    return {
        "id": task.id,
        "state": task.state,
        "result": task.result if task.ready() else None,
    }
