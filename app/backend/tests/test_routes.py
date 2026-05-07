"""
Backend tests — covers all routes.
Database interactions are mocked to avoid requiring PostgreSQL.
"""
import os
import uuid
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.backend.main import app
os.environ["TESTING"] = "1"

# ────────────────────────────────────────────────────────────────
# Fake SQLAlchemy-like Result + Session
# ────────────────────────────────────────────────────────────────

class FakeResult:
    def __init__(self, value=None, list_value=None):
        self.value = value
        self.list_value = list_value or []

    def scalar_one_or_none(self):
        return self.value

    def scalars(self):
        return self

    def all(self):
        return self.list_value

class FakeSession:
    """Fake session configurable par test."""
    def __init__(self):
        self._execute_result = FakeResult()
        self._get_result = None

    def set_execute_result(self, result):
        self._execute_result = result

    def set_get_result(self, result):
        self._get_result = result

    async def execute(self, *args, **kwargs):
        return self._execute_result

    async def get(self, *args, **kwargs):
        return self._get_result

    async def delete(self, *args, **kwargs):
        return None

    async def commit(self):
        return None

    async def refresh(self, *args, **kwargs):
        return None

    async def close(self):
        return None

# ────────────────────────────────────────────────────────────────
# Override FastAPI DB dependency
# ────────────────────────────────────────────────────────────────

TEST_SESSION = FakeSession()

@pytest.fixture(scope="session", autouse=True)
def override_db():
    from app.backend.main import app
    from app.backend.db.session import get_db

    async def fake_get_db():
        yield TEST_SESSION

    app.dependency_overrides[get_db] = fake_get_db
    yield
    app.dependency_overrides.clear()

# ────────────────────────────────────────────────────────────────
# Client fixture
# ────────────────────────────────────────────────────────────────
@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac

# ────────────────────────────────────────────────────────────────
# TESTS
# ────────────────────────────────────────────────────────────────

# Health
@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

# Waste
@pytest.mark.asyncio
async def test_list_waste_types_returns_list(client: AsyncClient):
    response = await client.get("/api/v1/waste/types")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

@pytest.mark.asyncio
async def test_list_waste_infos_returns_list(client: AsyncClient):
    response = await client.get("/api/v1/waste/infos")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

# Images
@pytest.mark.asyncio
async def test_upload_invalid_file_type(client: AsyncClient):
    response = await client.post(
        "/api/v1/images/upload",
        params={"user_id": str(uuid.uuid4())},
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400

@pytest.mark.asyncio
async def test_upload_file_too_large(client: AsyncClient):
    large_content = b"x" * (11 * 1024 * 1024)
    response = await client.post(
        "/api/v1/images/upload",
        params={"user_id": str(uuid.uuid4())},
        files={"file": ("big.jpg", large_content, "image/jpeg")},
    )
    assert response.status_code == 400

@pytest.mark.asyncio
async def test_get_nonexistent_image(client: AsyncClient):
    response = await client.get("/api/v1/images/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_delete_nonexistent_image(client: AsyncClient):
    response = await client.delete(
        "/api/v1/images/00000000-0000-0000-0000-000000000000"
    )
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_get_image_file_nonexistent(client: AsyncClient):
    response = await client.get(
        "/api/v1/images/00000000-0000-0000-0000-000000000000/file"
    )
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_get_all_images_returns_list(client: AsyncClient):
    response = await client.get("/api/v1/images/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

# Predictions
@pytest.mark.asyncio
async def test_get_nonexistent_prediction(client: AsyncClient):
    response = await client.get(
        "/api/v1/predictions/00000000-0000-0000-0000-000000000000"
    )
    assert response.status_code == 404

# Users — register
@pytest.mark.asyncio
async def test_register_missing_fields(client: AsyncClient):
    response = await client.post("/api/v1/users/register", json={"first_name": "Marie"})
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_register_password_too_short(client: AsyncClient):
    response = await client.post(
        "/api/v1/users/register",
        json={
            "first_name": "Marie",
            "last_name": "Dupont",
            "email": "marie@test.fr",
            "password": "short",
            "role": "user"
        },
    )
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient):
    # Simule un utilisateur déjà existant
    TEST_SESSION.set_execute_result(FakeResult(value=MagicMock()))

    payload = {
        "first_name": "Marie",
        "last_name": "Dupont",
        "email": "marie@test.fr",
        "password": "pass1234",
        "role": "user"
    }

    response = await client.post("/api/v1/users/register", json=payload)

    assert response.status_code == 409

# Users — login
@pytest.mark.asyncio
async def test_login_missing_fields(client: AsyncClient):
    response = await client.post("/api/v1/users/login", json={"email": "test@test.fr"})
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_login_wrong_credentials(client: AsyncClient):
    TEST_SESSION.set_execute_result(FakeResult(value=None))

    response = await client.post(
        "/api/v1/users/login",
        json={"email": "nobody@test.fr", "password": "wrongpass"},
    )

    assert response.status_code == 401

# Users — logout
@pytest.mark.asyncio
async def test_logout(client: AsyncClient):
    response = await client.post("/api/v1/users/logout")
    assert response.status_code == 200
    assert response.json()["message"] == "Logged out successfully"

# Users — get
@pytest.mark.asyncio
async def test_get_nonexistent_user(client: AsyncClient):
    TEST_SESSION.set_get_result(None)

    response = await client.get("/api/v1/users/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_list_users_returns_list(client: AsyncClient):
    TEST_SESSION.set_execute_result(FakeResult(list_value=[]))

    response = await client.get("/api/v1/users/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

# Users — update
@pytest.mark.asyncio
async def test_update_nonexistent_user(client: AsyncClient):
    TEST_SESSION.set_get_result(None)

    response = await client.put(
        "/api/v1/users/00000000-0000-0000-0000-000000000000",
        json={"first_name": "Nouveau"},
    )
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_update_user_password_too_short(client: AsyncClient):
    user_id = str(uuid.uuid4())
    mock_user = MagicMock()
    mock_user.id = uuid.UUID(user_id)
    mock_user.first_name = "Marie"
    mock_user.last_name = "Dupont"
    mock_user.email = "marie@test.fr"
    mock_user.role = "user"

    TEST_SESSION.set_get_result(mock_user)

    response = await client.put(
        f"/api/v1/users/{user_id}",
        json={"password": "short"},
    )

    assert response.status_code == 400

# Users — delete
@pytest.mark.asyncio
async def test_delete_nonexistent_user(client: AsyncClient):
    TEST_SESSION.set_get_result(None)

    response = await client.delete("/api/v1/users/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404

# Users — history
@pytest.mark.asyncio
async def test_get_history_nonexistent_user(client: AsyncClient):
    TEST_SESSION.set_get_result(None)
    response = await client.get(
        "/api/v1/users/00000000-0000-0000-0000-000000000000/history"
    )
    assert response.status_code == 404
