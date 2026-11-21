"""Tests verifying confidence gating, strict entity checks, and 'please speak again' errors."""

import io
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from app.auth import get_current_user
from app.database import get_session
from app.main import app
from app.models.expense import Expense
from app.models.settlement import Settlement
from app.models.user import User
from app.schemas.ai import (
    ExtractedItem,
    IntentType,
    SettleExtraction,
    SplitTypeHint,
    StageOneOutput,
    StageTwoResolution,
)

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DB_URL, echo=False, future=True)
async_session_maker_test = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)


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
async def client_with_user(test_db_session: AsyncSession):
    user = User(
        id=uuid.uuid4(),
        email="reliable@example.com",
        name="Satendra",
        phone="+919999900099",
    )
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
        yield c, user, test_db_session

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_unclear_audio_returns_speak_again_and_writes_nothing(client_with_user):
    client, _, session = client_with_user

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text="[UNCLEAR]")
    )

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        resp = await client.post(
            "/voice/audio",
            files={"file": ("noisy.m4a", io.BytesIO(b"noisy bytes"), "audio/m4a")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ERROR"
    assert data["error_code"] == "UNCLEAR_AUDIO"
    assert data["count"] == 0
    assert data["expenses"] == []
    assert "speak again" in data["spoken_confirmation"].lower()

    expenses = (await session.execute(select(Expense))).scalars().all()
    assert expenses == []


@pytest.mark.asyncio
async def test_low_confidence_stage1_rejected_never_added_to_self(client_with_user):
    client, _, session = client_with_user

    mock_stage1 = StageOneOutput(
        intent=IntentType.CREATE,
        confidence="LOW",
        ambiguity_reason="Audio was muffled around amount and friend name",
        is_complex=False,
        items=[
            ExtractedItem(
                description="Food",
                amount=200.0,
                participants_raw=["me"],
                split_type=SplitTypeHint.EQUAL,
            )
        ],
        spoken_confirmation="Sound was not clear. Please speak again clearly.",
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_stage1.model_dump_json())
    )

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        resp = await client.post("/voice/text", json={"text": "umm 200 something"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ERROR"
    assert data["error_code"] == "LOW_CONFIDENCE"
    assert data["count"] == 0
    assert data["expenses"] == []
    assert "speak again" in data["spoken_confirmation"].lower()

    expenses = (await session.execute(select(Expense))).scalars().all()
    assert expenses == []


@pytest.mark.asyncio
async def test_unknown_friend_rejected_not_logged_to_self(client_with_user):
    client, _, session = client_with_user

    mock_stage1 = StageOneOutput(
        intent=IntentType.CREATE,
        confidence="HIGH",
        is_complex=False,
        items=[
            ExtractedItem(
                description="Lunch",
                amount=200.0,
                payer_raw="me",
                participants_raw=["Rishabh"],
                split_type=SplitTypeHint.EQUAL,
            )
        ],
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_stage1.model_dump_json())
    )

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        resp = await client.post("/voice/text", json={"text": "add 200 on rishabh"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ERROR"
    assert data["error_code"] == "UNKNOWN_FRIEND"
    assert data["count"] == 0
    assert data["expenses"] == []
    assert "rishabh" in data["spoken_confirmation"].lower()
    assert "speak again" in data["spoken_confirmation"].lower()

    expenses = (await session.execute(select(Expense))).scalars().all()
    assert expenses == []


@pytest.mark.asyncio
async def test_tied_duplicate_friends_rejected_as_ambiguous(client_with_user):
    client, _, session = client_with_user

    # Add two friends with the same name "Rishabh" and no distinguishing alias used
    await client.post(
        "/friends",
        json={"friend_phone": "+919800011111", "friend_name": "Rishabh", "nickname": "Rishabh Office"},
    )
    await client.post(
        "/friends",
        json={"friend_phone": "+919800022222", "friend_name": "Rishabh", "nickname": "Rishabh Flat"},
    )

    mock_stage1 = StageOneOutput(
        intent=IntentType.CREATE,
        confidence="HIGH",
        is_complex=False,
        items=[
            ExtractedItem(
                description="Lunch",
                amount=200.0,
                payer_raw="me",
                participants_raw=["Rishabh"],
                split_type=SplitTypeHint.EQUAL,
            )
        ],
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_stage1.model_dump_json())
    )

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        resp = await client.post("/voice/text", json={"text": "add 200 on rishabh"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "AMBIGUOUS"
    assert data["is_ambiguous"] is True
    assert data["error_code"] == "AMBIGUOUS_FRIEND"
    assert data["count"] == 0
    assert "rishabh office" in data["clarification_prompt"].lower()
    assert "rishabh flat" in data["clarification_prompt"].lower()

    expenses = (await session.execute(select(Expense))).scalars().all()
    assert expenses == []


@pytest.mark.asyncio
async def test_unknown_group_and_invalid_amount_rejected(client_with_user):
    client, _, session = client_with_user

    # Unknown group
    mock_bad_group = StageOneOutput(
        intent=IntentType.CREATE,
        confidence="HIGH",
        is_complex=False,
        items=[
            ExtractedItem(
                description="Dinner",
                amount=500.0,
                participants_raw=["me"],
                group_raw="Nonexistent Group",
                split_type=SplitTypeHint.EQUAL,
            )
        ],
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_bad_group.model_dump_json())
    )

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        resp = await client.post("/voice/text", json={"text": "add 500 in Nonexistent Group"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "ERROR"
    assert resp.json()["error_code"] == "UNKNOWN_GROUP"
    assert resp.json()["count"] == 0

    # Zero/negative amount
    mock_zero_amt = StageOneOutput(
        intent=IntentType.CREATE,
        confidence="HIGH",
        is_complex=False,
        items=[
            ExtractedItem(
                description="Tea",
                amount=0.0,
                participants_raw=["me"],
                split_type=SplitTypeHint.EQUAL,
            )
        ],
    )
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_zero_amt.model_dump_json())
    )

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        resp2 = await client.post("/voice/text", json={"text": "add tea"})

    assert resp2.status_code == 200
    assert resp2.json()["status"] == "ERROR"
    assert resp2.json()["error_code"] == "INVALID_AMOUNT"

    expenses = (await session.execute(select(Expense))).scalars().all()
    assert expenses == []


@pytest.mark.asyncio
async def test_multi_create_atomic_no_partial_commit(client_with_user):
    client, _, session = client_with_user

    mock_multi = StageOneOutput(
        intent=IntentType.MULTI_CREATE,
        confidence="HIGH",
        is_complex=False,
        items=[
            ExtractedItem(description="Coffee", amount=100.0, participants_raw=["me"]),
            ExtractedItem(description="Snack", amount=200.0, participants_raw=["GhostFriend"]),
        ],
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_multi.model_dump_json())
    )

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        resp = await client.post("/voice/text", json={"text": "100 coffee and 200 on GhostFriend"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "ERROR"
    assert resp.json()["error_code"] == "UNKNOWN_FRIEND"
    assert resp.json()["count"] == 0

    # Neither item 1 nor item 2 was written
    expenses = (await session.execute(select(Expense))).scalars().all()
    assert expenses == []


@pytest.mark.asyncio
async def test_settle_unknown_friend_does_not_pick_balance_fallback(client_with_user):
    client, caller, session = client_with_user

    # Seed an existing friend + balance so balances is non-empty
    f_res = (
        await client.post(
            "/friends",
            json={"friend_phone": "+919877777777", "friend_name": "Sahaj", "nickname": "Sahaj"},
        )
    ).json()
    sahaj_id = f_res["friend_id"]
    await client.post(
        "/expenses",
        json={
            "description": "Dinner",
            "amount": "500.00",
            "payer_id": str(caller.id),
            "splits": [{"user_id": sahaj_id, "owed_amount": "500.00"}],
        },
    )

    mock_settle = StageOneOutput(
        intent=IntentType.SETTLE,
        confidence="HIGH",
        is_complex=False,
        settle_item=SettleExtraction(payee_raw="UnknownPerson", amount=200.0),
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_settle.model_dump_json())
    )

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        resp = await client.post("/voice/text", json={"text": "I paid UnknownPerson 200"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "ERROR"
    assert resp.json()["error_code"] == "UNKNOWN_FRIEND"

    settlements = (await session.execute(select(Settlement))).scalars().all()
    assert settlements == []


@pytest.mark.asyncio
async def test_stage2_medium_confidence_is_blocked(client_with_user):
    client, _, session = client_with_user

    mock_s1 = StageOneOutput(
        intent=IntentType.CREATE,
        confidence="HIGH",
        is_complex=True,
        items=[ExtractedItem(description="Dinner", amount=400.0, participants_raw=["me"])],
    )
    mock_s2 = StageTwoResolution(
        action_type="CREATE",
        confidence="MEDIUM",
        description="Dinner",
        amount=Decimal("400.00"),
        clarification_question="Not sure which group or friends to split with. Please speak again.",
    )
    mock_client = MagicMock()

    async def mock_gen(model, **kwargs):
        if "flash" in model:
            return MagicMock(text=mock_s1.model_dump_json())
        return MagicMock(text=mock_s2.model_dump_json())

    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_gen)

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        resp = await client.post("/voice/text", json={"text": "Add 400 dinner yesterday"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "ERROR"
    assert resp.json()["error_code"] == "LOW_CONFIDENCE"
    assert resp.json()["count"] == 0

    expenses = (await session.execute(select(Expense))).scalars().all()
    assert expenses == []
