from fastapi import FastAPI

from api.routers.transcription_router import router as transcription_router
from api.routers.tasks_router import router as tasks_router
from api.routers.operators_router import router as operators_router

app = FastAPI()
app.include_router(transcription_router)
app.include_router(tasks_router)
app.include_router(operators_router)


@app.get("/")
async def welcome():
    return "Welcome"
