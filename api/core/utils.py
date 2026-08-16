import asyncio
import gc
import re
import logging
from pathlib import Path

import torch

from api.core.config import settings

FILENAME_PATTERN = re.compile(
    r"^(?P<unix_ts>\d+)\.(?P<call_id>[^-]+)-(?P<operator_ext>\d{4})-(?P<random>.+)$"
)

logger = logging.getLogger(__name__)


def run_async_coro(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def release_gpu_memory() -> None:
    logger.info("Memory cleanup")
    gc.collect()
    if torch.cuda.is_available() and settings.DEVICE == "cuda":
        torch.cuda.empty_cache()


def parse_call_metadata(filename: str) -> dict | None:
    stem = Path(filename).stem
    match = FILENAME_PATTERN.match(stem)
    if not match:
        return 
