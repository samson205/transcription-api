from fastapi import FastAPI

from api.transcription.router import router as t_router

app = FastAPI()
app.include_router(t_router)


@app.get("/")
async def welcome():
    return "Welcome"
