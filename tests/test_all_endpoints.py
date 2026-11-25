"""Verification test hitting every single registered endpoint family."""

import io
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from app.auth import get_current_user
from app.database import get_session
from app.main import app
from app.models.user import User
from app.schemas.ai import ExtractedItem, IntentType, SplitTypeHint, StageOneOutput

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(TEST_DB_URL, echo=False, future=True)
async_session_maker_test = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)


@pytest.fixture(scope="session", autouse=True)
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def test_db_session():
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with async_session_maker_test() as session:
        yield session
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


@pytest.fixture
async def client(test_db_session: AsyncSession):
    user = User(id=uuid.uuid4(), email="lead@example.com", name="Lead User", phone="+919000011111")
    test_db_session.add(user)
    await test_db_session.commit()
    await test_db_session.refresh(user)

    async def override_get_session():
        yield test_db_session

    async def override_get_current_user():
        return user

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_current_user] = override_get_current_user

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, user

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_all_api_endpoints_end_to_end(client):
    c, user = client

    # 1. Health
    h = await c.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "healthy"

    # 2. Friends: POST & GET
    f_res = await c.post("/friends", json={"friend_name": "Friend A", "friend_email": "frienda@test.com", "friend_phone": "+919888800000"})
    assert f_res.status_code == 201
    friend_id = f_res.json()["friend_id"]

    friends_list = await c.get("/friends")
    assert friends_list.status_code == 200
    assert len(friends_list.json()) >= 1

    # 3. Groups: POST, GET, GET by ID, POST members, POST join
    g_res = await c.post("/groups", json={"name": "Audit Trip"})
    assert g_res.status_code == 201
    group_id = g_res.json()["id"]
    invite_code = g_res.json()["invite_code"]

    groups_list = await c.get("/groups")
    assert groups_list.status_code == 200
    assert len(groups_list.json()) >= 1

    g_detail = await c.get(f"/groups/{group_id}")
    assert g_detail.status_code == 200
    assert g_detail.json()["name"] == "Audit Trip"

    add_m = await c.post(f"/groups/{group_id}/members", json={"identifier": str(friend_id)})
    assert add_m.status_code == 201

    join_res = await c.post(f"/groups/join/{invite_code}")
    assert join_res.status_code == 200

    # 4. Expenses: POST, GET, GET by ID, PATCH, DELETE
    exp_res = await c.post(
        "/expenses",
        json={
            "description": "Lunch Buffet",
            "amount": "1200.00",
            "payer_id": str(user.id),
            "group_id": group_id,
            "splits": [
                {"user_id": str(user.id), "owed_amount": "600.00"},
                {"user_id": str(friend_id), "owed_amount": "600.00"},
            ],
        },
    )
    assert exp_res.status_code == 201
    expense_id = exp_res.json()["id"]

    exp_list = await c.get("/expenses")
    assert exp_list.status_code == 200
    assert len(exp_list.json()) >= 1

    exp_single = await c.get(f"/expenses/{expense_id}")
    assert exp_single.status_code == 200
    assert exp_single.json()["description"] == "Lunch Buffet"

    exp_patch = await c.patch(f"/expenses/{expense_id}", json={"description": "Lunch Buffet & Chai", "amount": 1300.00})
    assert exp_patch.status_code == 200
    assert exp_patch.json()["description"] == "Lunch Buffet & Chai"

    # 5. Activity Feeds: Group & Bilateral
    grp_feed = await c.get(f"/groups/{group_id}/expenses")
    assert grp_feed.status_code == 200
    assert len(grp_feed.json()["timeline"]) >= 1

    friend_feed = await c.get(f"/friends/{friend_id}/expenses")
    assert friend_feed.status_code == 200
    assert "frienda" in friend_feed.json()["friend_name"].lower() or "friend a" in friend_feed.json()["friend_name"].lower()

    # 6. Balances
    bal = await c.get("/balances")
    assert bal.status_code == 200
    assert Decimal(str(bal.json()["total_owed_to_you"])) > Decimal("0.00")

    # 7. Settlements: POST, GET, DELETE
    set_res = await c.post(
        "/settlements",
        json={
            "payee_id": str(friend_id),
            "amount": "650.00",
            "payment_method": "UPI",
            "notes": "Full settlement",
        },
    )
    assert set_res.status_code == 201
    settlement_id = set_res.json()["id"]

    set_list = await c.get("/settlements")
    assert set_list.status_code == 200
    assert len(set_list.json()) >= 1

    set_del = await c.delete(f"/settlements/{settlement_id}")
    assert set_del.status_code == 204

    # 8. Delete Expense & verify Activity Log
    exp_del = await c.delete(f"/expenses/{expense_id}")
    assert exp_del.status_code == 204

    act_res = await c.get("/activity")
    assert act_res.status_code == 200
    act_events = [item["event_type"] for item in act_res.json()]
    assert "EXPENSE_DELETED" in act_events
    assert "SETTLEMENT_DELETED" in act_events

    # 9. Voice Text & Voice Audio
    mock_stage1 = StageOneOutput(
        intent=IntentType.CREATE,
        is_complex=False,
        items=[
            ExtractedItem(
                description="Cold Drink",
                amount=80.0,
                payer_raw="me",
                participants_raw=["me"],
                split_type=SplitTypeHint.EQUAL,
            )
        ],
        spoken_confirmation="Added ₹80 for cold drink.",
    )
    mock_client = MagicMock()

    async def mock_gen(*args, **kwargs):
        contents = kwargs.get("contents", "")
        # If audio transcription prompt
        if isinstance(contents, list) and any("Transcribe" in str(x) for x in contents):
            return MagicMock(text="Add 80 for cold drink")
        return MagicMock(text=mock_stage1.model_dump_json())

    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_gen)

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        voice_t = await c.post("/voice/text", json={"text": "Add 80 for cold drink"})
        assert voice_t.status_code == 200
        assert voice_t.json()["status"] == "SUCCESS"

        audio_bytes = io.BytesIO(b"dummy audio content")
        voice_a = await c.post(
            "/voice/audio",
            files={"file": ("test.m4a", audio_bytes, "audio/m4a")},
        )
        assert voice_a.status_code == 200
        assert voice_a.json()["status"] == "SUCCESS"
        assert voice_a.json()["staged_items"] == []
