import asyncio

from api.core.celery import celery
from api.core.dependencies import get_conversation_service


@celery.task
def transcribe_task(path: str):
    return asyncio.run(_run_transcribe(path))


async def _run_transcribe(path: str):
    service = get_conversation_service()
    result = await service.process(path)
    return result.model_dump()
