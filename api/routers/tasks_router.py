from fastapi import APIRouter

from api.core.schemas import TaskResponse
from api.services.task_service import TaskService

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    task = TaskService.status(task_id)

    return {
        "task_id": task.id,
        "state": task.state,
        "result": task.result if task.ready() else None,
    }
