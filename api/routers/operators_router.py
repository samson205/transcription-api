from fastapi import APIRouter

router = APIRouter(prefix="/operators", tags=["operators"])


@router.post("/")
async def create_operator(

):
    pass
