"""
Backend tests — health check, image upload, waste info endpoints.
Uses httpx AsyncClient to test FastAPI routes without spinning up a real server.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.backend.main import app


@pytest.fixture
async def client():
    """Async test client for FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    """Health endpoint should return 200 with status ok."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_list_waste_types(client: AsyncClient):
    """Waste types endpoint should return a list."""
    response = await client.get("/api/v1/waste/types")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_upload_invalid_file_type(client: AsyncClient):
    """Upload with wrong content type should return 400."""
    response = await client.post(
        "/api/v1/images/upload",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_nonexistent_image(client: AsyncClient):
    """Fetching a non-existent image UUID should return 404."""
    response = await client.get(
        "/api/v1/images/00000000-0000-0000-0000-000000000000"
    )
    assert response.status_code == 404
