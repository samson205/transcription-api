from fastapi import APIRouter

from api.core.schemas import TaskResponse
from api.services.task_service import TaskService

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    task = TaskService.status(task_id)
    
    task_result = None
    if task.ready():
        if task.state == "SUCCESS":
            task_result = task.result
        elif task.state == "FAILURE":
            task_result = {"error": str(task.result)}
    return {
        "task_id": task.id,
        "state": task.state,
        "result": task_result,
    }
