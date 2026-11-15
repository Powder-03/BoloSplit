"""Unit tests verifying Firebase Authentication and shadow account claiming."""

from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from app.database import get_session
from app.main import app
from app.models.user import User

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(TEST_DB_URL, echo=False, future=True)
async_session_maker_test = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)


@pytest.fixture(scope="session", autouse=True)
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def test_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with async_session_maker_test() as session:
        yield session
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


@pytest.fixture
async def client_unauthed(test_db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    async def override_get_session():
        yield test_db_session

    app.dependency_overrides[get_session] = override_get_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_firebase_token_creates_new_user(client_unauthed: AsyncClient):
    """Verify that a valid Firebase Google/Phone token provisions a new user with firebase_uid."""
    mock_decoded = {
        "uid": "fb_google_12345",
        "email": "googleuser@gmail.com",
        "name": "Google User",
        "phone_number": "+919876543210",
    }

    with patch("app.auth.init_firebase", return_value=True), \
         patch("app.auth.fb_auth.verify_id_token", return_value=mock_decoded):
        resp = await client_unauthed.get(
            "/balances",
            headers={"Authorization": "Bearer fake_valid_firebase_jwt"},
        )

    assert resp.status_code == 200
    assert resp.json()["total_owed_to_you"] == "0.00"


@pytest.mark.asyncio
async def test_firebase_claims_shadow_user_by_phone(client_unauthed: AsyncClient, test_db_session: AsyncSession):
    """Verify that when a friend who was added as a shadow user logs in, their account is claimed."""
    # Pre-seed shadow user (added by a friend earlier by phone)
    shadow = User(
        email="user_919999900000@bolosplit.app",
        name="Friend 0000",
        phone="+919999900000",
    )
    test_db_session.add(shadow)
    await test_db_session.commit()
    await test_db_session.refresh(shadow)

    mock_decoded = {
        "uid": "fb_phone_99999",
        "phone_number": "+919999900000",
        "name": "Sahaj Sharma",
    }

    with patch("app.auth.init_firebase", return_value=True), \
         patch("app.auth.fb_auth.verify_id_token", return_value=mock_decoded):
        resp = await client_unauthed.get(
            "/balances",
            headers={"Authorization": "Bearer fake_phone_otp_jwt"},
        )

    assert resp.status_code == 200

    # Verify shadow account was updated with verified name and firebase_uid
    updated = (await test_db_session.get(User, shadow.id))
    assert updated.firebase_uid == "fb_phone_99999"
    assert updated.name == "Sahaj Sharma"


@pytest.mark.asyncio
async def test_firebase_invalid_token_returns_401(client_unauthed: AsyncClient):
    """Verify that an invalid Firebase token is rejected with 401 Unauthorized."""
    with patch("app.auth.init_firebase", return_value=True), \
         patch("app.auth.fb_auth.verify_id_token", side_effect=Exception("Signature expired")):
        resp = await client_unauthed.get(
            "/balances",
            headers={"Authorization": "Bearer bad_token"},
        )

    assert resp.status_code == 401
    assert "Invalid or expired Firebase ID token" in resp.json()["detail"]
