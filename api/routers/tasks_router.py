from fastapi import APIRouter

from api.schemas.task import TaskResponse
from api.services.task_service import TaskService

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    task = TaskService.status(task_id)

    error_message = str(task.result) if task.state == "FAILURE" else None
    return {
        "task_id": task.id,
        "state": task.state,
        "error": error_message,
    }
