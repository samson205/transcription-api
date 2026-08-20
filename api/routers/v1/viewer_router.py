# api/routers/viewer_router.py
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.core.dependencies import get_conversation_service
from api.services.conversation_service import ConversationService

router = APIRouter(prefix="/viewer", tags=["Viewer"])
templates = Jinja2Templates(directory="api/templates")


@router.get("/", response_class=HTMLResponse)
async def list_conversations(
    request: Request,
    page: int = 1,
    service: ConversationService = Depends(get_conversation_service),
):
    conversations = await service.get_all(page, 20)
    return templates.TemplateResponse(
        request=request,
        name="conversations_list.html",
        context={"request": request, "conversations": conversations, "page": page},
    )


@router.get("/{conversation_id}", response_class=HTMLResponse)
async def view_conversation(
    request: Request,
    conversation_id: int,
    service: ConversationService = Depends(get_conversation_service),
):
    try:
        conversation = await service.get_by_id(conversation_id)
    except ValueError:
        return templates.TemplateResponse(
            request=request,
            name="not_found.html",
            context={"request": request, "conversation_id": conversation_id},
            status_code=404,
        )
    return templates.TemplateResponse(
        request=request, name="conversation_detail.html", context={"request": request, "conversation": conversation}
    )
