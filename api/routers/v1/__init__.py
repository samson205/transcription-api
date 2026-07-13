from fastapi import APIRouter

from api.routers.v1.conversations_router import router as conversation_router
from api.routers.v1.operators_router import router as operators_router

router = APIRouter(prefix="/api/v1")
router.include_router(conversation_router)
router.include_router(operators_router)
