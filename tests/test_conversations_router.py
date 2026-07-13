from unittest.mock import AsyncMock
from datetime import datetime

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from api.core.dependencies import get_conversation_service
from api.models.conversation_model import Conversation
from api.models.enums import ProcessingStatus


@pytest.fixture
def mock_conversation_service():
    service = AsyncMock()
    app.dependency_overrides[get_conversation_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_conversation_service, None)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestGetConversation:
    @pytest.mark.parametrize(
        "id,status",
        [
            (1, ProcessingStatus.SUCCESS), (2, ProcessingStatus.PENDING), (3, ProcessingStatus.PROCESSING)
        ]
    )
    async def test_returns_conversation(self, client, mock_conversation_service, id, status):
        mock_conversation_service.get_by_id.return_value = Conversation(
            id=id,
            filename="call.mp3",
            status=status,
            error_message=None,
            language="ru",
            duration=10.0,
            created_at=datetime.fromisoformat("2026-07-13T12:00:00Z"),
            segments=[]
        )

        response = await client.get(f"/api/v1/conversations/{id}")

        assert response.status_code == 200
        assert response.json()["id"] == id
        assert response.json()["status"] == status
        assert response.json()["filename"] == "call.mp3"

    async def test_conversation_not_found(self, client, mock_conversation_service):
        mock_conversation_service.get_by_id.side_effect = ValueError("Conversation not found")

        response = await client.get("/api/v1/conversations/1")

        assert response.status_code == 404