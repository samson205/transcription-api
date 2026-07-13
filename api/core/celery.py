from celery import Celery
from celery.signals import worker_process_init

from api.core.config import settings
from api.core.logging import setup_logging

celery = Celery("transcription", broker=settings.REDIS_URL, backend=settings.REDIS_URL)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
)


@worker_process_init.connect
def _init_worker_logging(**kwargs):
    setup_logging()


import api.tasks.transcription_tasks
import api.tasks.operator_tasks
