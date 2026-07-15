import asyncio
import gc
import logging

import torch

from api.core.config import settings

logger = logging.getLogger(__name__)


def run_async_coro(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def release_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available() and settings.DEVICE == "cuda":
        torch.cuda.empty_cache()
        logger.info("GPU memory cleanup")
