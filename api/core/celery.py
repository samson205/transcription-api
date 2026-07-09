from celery import Celery

from api.core.config import settings

celery = Celery(
    "transcription",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
)

import api.tasks.transcription_tasks
import api.tasks.operator_tasks
