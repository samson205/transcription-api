from celery.result import AsyncResult

from api.core.celery import celery
from api.tasks.transcription import transcribe_task


class TaskService:
    @staticmethod
    def transcribe_task(path: str):
        task = transcribe_task.delay(path)
        return task.id
    
    @staticmethod
    def status(task_id: str):
        return AsyncResult(task_id, app=celery)
    