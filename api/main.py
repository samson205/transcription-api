from fastapi import FastAPI

from api.routers.v1 import router as v1_router
from api.core.logging import setup_logging

setup_logging()

app = FastAPI()
app.include_router(v1_router)


@app.get("/")
async def welcome():
    return "Welcome"
