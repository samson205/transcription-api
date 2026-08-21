from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routers.v1 import router as v1_router
from api.core.config import settings
from api.core.logging import setup_logging

setup_logging()

app = FastAPI()
app.mount("/audio", StaticFiles(directory=str(settings.AUDIO_DIR)), name="audio")
app.include_router(v1_router)


@app.get("/")
async def welcome():
    return "Welcome"
