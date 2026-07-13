from unittest.mock import AsyncMock
from datetime import datetime

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from api.core.dependencies import get_operator_service
from api.models.enums import ProcessingStatus
from api.models.operator_model import Operator


@pytest.fixture
def mock_operator_service():
    service = AsyncMock()
    app.dependency_overrides[get_operator_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_operator_service, None)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestGetOperator:
    @pytest.mark.parametrize(
        "id,status",
        [
            (1, ProcessingStatus.SUCCESS), (2, ProcessingStatus.PENDING), (3, ProcessingStatus.PROCESSING)
        ]
    )
    async def test_returns_operator(self, client, mock_operator_service, id, status):
        mock_operator_service.get_by_id.return_value = Operator(
            id=id,
            name="Alex",
            status=status,
            error_message=None,
            created_at=datetime.fromisoformat("2026-07-13T12:00:00Z")
        )

        response = await client.get(f"/api/v1/operators/{id}")

        assert response.status_code == 200
        assert response.json()["id"] == id
        assert response.json()["status"] == status

    async def test_operator_not_found(self, client, mock_operator_service):
        mock_operator_service.get_by_id.side_effect = ValueError("Operator not found")

        response = await client.get("/api/v1/operators/1")

        assert response.status_code == 404