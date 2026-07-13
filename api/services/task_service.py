from celery.result import AsyncResult

from api.core.celery import celery
from api.tasks.transcription_tasks import transcribe_task
from api.tasks.operator_tasks import extract_operator_embedding_task


class TaskService:
    @staticmethod
    def create_transcribe_task(conversation_id: int, path: str, original_filename: str):
        task = transcribe_task.apply_async(
            args=[conversation_id, path, original_filename]
        )
        return task.id

    @staticmethod
    def create_extract_operator_embedding_task(operator_id: int, file_path: str):
        task = extract_operator_embedding_task.delay(operator_id, file_path)
        return task.id

    @staticmethod
    def status(task_id: str):
        return AsyncResult(task_id, app=celery)
