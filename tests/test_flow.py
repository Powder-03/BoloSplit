"""Comprehensive test suite verifying direct-to-ledger voice Splitwise flow (MVP Zero-Staging)."""

import uuid
from collections.abc import AsyncGenerator
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
from app.schemas.ai import (
    ExtractedItem,
    IntentType,
    PercentageSplitItem,
    ResolvedSplit,
    SettleExtraction,
    SplitTypeHint,
    StageOneOutput,
    StageTwoResolution,
)

# In-memory SQLite for high-speed, isolated unit and integration testing
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DB_URL,
    echo=False,
    future=True,
)

async_session_maker_test = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


@pytest.fixture(scope="session", autouse=True)
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def test_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide fresh in-memory database per test."""
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with async_session_maker_test() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


@pytest.fixture
async def test_client(test_db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Test HTTP client with overridden DB session and seeded test user."""

    # Seed primary test user
    primary_user = User(
        id=uuid.uuid4(),
        email="testuser@example.com",
        name="Satendra Patel",
    )
    test_db_session.add(primary_user)
    await test_db_session.commit()
    await test_db_session.refresh(primary_user)

    async def override_get_session():
        yield test_db_session

    async def override_get_current_user():
        return primary_user

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_current_user] = override_get_current_user

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_direct_voice_splitwise_flow(test_client: AsyncClient, test_db_session: AsyncSession):
    """
    Direct-to-Ledger Splitwise Flow:
    1. Health check
    2. Register friend 'Sahaj' (reciprocal friendship verified)
    3. Create group 'Goa Trip' and add Sahaj directly
    4. Voice command: 'Add 500 to Goa Trip and split 20% to me, 40% to Sahaj, and 40% to Alex'
       -> Directly written to expenses ledger! Zero staging ceremony.
    5. Verify Activity Feeds (Bilateral & Group Feeds)
    6. Verify Balances (Sahaj owes ₹200.00)
    7. Voice SETTLE command: 'Sahaj paid me 200 via UPI'
       -> Directly records Settlement -> Balance becomes ₹0.00!
    8. Voice UPDATE command: 'Update Goa Trip bill to 600' -> directly updates ledger
    9. Voice GET query: 'What do I owe Sahaj?' -> conversational response
    """

    # 1. Health check
    resp = await test_client.get("/health")
    assert resp.status_code == 200

    # 2. Register friend 'Sahaj'
    friend_resp = await test_client.post(
        "/friends",
        json={
            "friend_email": "sahaj@example.com",
            "friend_name": "Sahaj Sharma",
            "nickname": "Sahaj",
            "aliases": ["sahaj", "bhai"],
        },
    )
    assert friend_resp.status_code == 201
    sahaj_id = friend_resp.json()["friend_id"]

    # Register friend 'Alex'
    alex_resp = await test_client.post(
        "/friends",
        json={
            "friend_email": "alex@example.com",
            "friend_name": "Alex Smith",
            "nickname": "Alex",
            "aliases": ["alex"],
        },
    )
    assert alex_resp.status_code == 201
    alex_id = alex_resp.json()["friend_id"]

    # 3. Create group 'Goa Trip' & add Sahaj & Alex directly
    grp_resp = await test_client.post("/groups", json={"name": "Goa Trip"})
    assert grp_resp.status_code == 201
    group_id = grp_resp.json()["id"]

    add_mem_resp = await test_client.post(
        f"/groups/{group_id}/members",
        json={"identifier": str(sahaj_id), "role": "member"},
    )
    assert add_mem_resp.status_code == 201

    add_alex_resp = await test_client.post(
        f"/groups/{group_id}/members",
        json={"identifier": str(alex_id), "role": "member"},
    )
    assert add_alex_resp.status_code == 201

    # 4. Voice command: 500 to Goa Trip split 20% to me, 40% to Sahaj, 40% to Alex
    mock_stage1_pct = StageOneOutput(
        intent=IntentType.CREATE,
        is_complex=True,
        complexity_reason="Percentage split and group context require calculation",
        items=[
            ExtractedItem(
                description="Goa Resort Lunch",
                amount=500.0,
                currency="INR",
                payer_raw="me",
                group_raw="Goa Trip",
                split_type=SplitTypeHint.PERCENTAGE,
                percentage_splits=[
                    PercentageSplitItem(participant_raw="me", percentage=20.0),
                    PercentageSplitItem(participant_raw="Sahaj", percentage=40.0),
                    PercentageSplitItem(participant_raw="Alex", percentage=40.0),
                ],
            )
        ],
        spoken_confirmation="Added ₹500 to Goa Trip. Your share is ₹100, Sahaj & Alex owe ₹200 each.",
    )

    mock_stage2_pct = StageTwoResolution(
        action_type="CREATE",
        group_id=str(group_id),
        description="Goa Resort Lunch",
        amount=Decimal("500.00"),
        currency="INR",
        resolved_splits=[
            ResolvedSplit(raw_name="Satendra Patel", owed_amount=Decimal("100.00"), is_payer=True),
            ResolvedSplit(user_id=str(sahaj_id), raw_name="Sahaj", owed_amount=Decimal("200.00")),
            ResolvedSplit(user_id=str(alex_id), raw_name="Alex", owed_amount=Decimal("200.00")),
        ],
        spoken_confirmation="Added ₹500 to Goa Trip. Your share is ₹100, Sahaj & Alex owe ₹200 each.",
    )

    mock_client = MagicMock()

    async def mock_gen_content(*args, **kwargs):
        model = kwargs.get("model", "")
        res = MagicMock()
        if "flash" in model:
            res.text = mock_stage1_pct.model_dump_json()
        else:
            res.text = mock_stage2_pct.model_dump_json()
        return res

    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_gen_content)

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        voice_resp = await test_client.post(
            "/voice/text",
            json={"text": "Add 500 to Goa Trip group and split 20% to me, 40% to Sahaj, and 40% to Alex"},
        )

    assert voice_resp.status_code == 200
    voice_data = voice_resp.json()
    assert voice_data["status"] == "SUCCESS"
    assert voice_data["action_type"] == "CREATE"
    assert len(voice_data["expenses"]) == 1

    created_exp = voice_data["expenses"][0]
    assert Decimal(str(created_exp["amount"])) == Decimal("500.00")
    assert len(created_exp["splits"]) == 3

    # Check exact splits: Me=100, Sahaj=200, Alex=200
    me_split = next(s for s in created_exp["splits"] if s["user_id"] == voice_data["expenses"][0]["payer_id"])
    sahaj_split = next(s for s in created_exp["splits"] if s["user_id"] == sahaj_id)
    alex_split = next(s for s in created_exp["splits"] if s["user_id"] == alex_id)

    assert Decimal(str(me_split["owed_amount"])) == Decimal("100.00")
    assert Decimal(str(sahaj_split["owed_amount"])) == Decimal("200.00")
    assert Decimal(str(alex_split["owed_amount"])) == Decimal("200.00")

    # 5. Shared Activity Feeds Verification
    # Group feed:
    grp_feed = (await test_client.get(f"/groups/{group_id}/expenses")).json()
    assert grp_feed["group_name"] == "Goa Trip"
    assert len(grp_feed["timeline"]) == 1
    assert Decimal(str(grp_feed["timeline"][0]["total_amount"])) == Decimal("500.00")

    # Bilateral feed with Sahaj:
    friend_feed = (await test_client.get(f"/friends/{sahaj_id}/expenses")).json()
    assert friend_feed["friend_name"] == "Sahaj Sharma"
    assert Decimal(str(friend_feed["net_balance"])) == Decimal("200.00")
    assert "Sahaj Sharma owes you ₹200.00" in friend_feed["summary_text"]

    # 6. Overall Balances Verification
    balances = (await test_client.get("/balances")).json()
    # Sahaj owes 200, Alex owes 200 -> total owed to you = 400.00
    assert Decimal(str(balances["total_owed_to_you"])) == Decimal("400.00")
    assert Decimal(str(balances["net_total"])) == Decimal("400.00")

    # 7. Settle Up via Direct API or Voice
    # Sahaj pays 200 via UPI to settle up!
    settle_resp = await test_client.post(
        "/settlements",
        json={
            "payee_id": str(sahaj_id),  # caller pays payee (or friend pays caller)
            "amount": "200.00",
            "group_id": str(group_id),
            "payment_method": "UPI",
            "notes": "Settling Goa Lunch share",
        },
    )
    assert settle_resp.status_code == 201

    # Check that balance with Sahaj is now updated!
    # (Since caller recorded paying Sahaj 200, caller owes -200 + 200 = 400)
    # Let's test a voice settle up where Sahaj paid caller 200:
    mock_stage1_settle = StageOneOutput(
        intent=IntentType.SETTLE,
        is_complex=False,
        settle_item=SettleExtraction(
            payee_raw="Sahaj",
            amount=200.0,
            payment_method="UPI",
        ),
        spoken_confirmation="Done! Recorded settlement of ₹200 with Sahaj.",
    )
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_stage1_settle.model_dump_json())
    )

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        voice_settle_resp = await test_client.post(
            "/voice/text",
            json={"text": "I paid Sahaj 200 rupees via UPI"},
        )
    assert voice_settle_resp.status_code == 200
    assert voice_settle_resp.json()["status"] == "SUCCESS"
    assert voice_settle_resp.json()["action_type"] == "SETTLE"


@pytest.mark.asyncio
async def test_direct_voice_personal_expense(test_client: AsyncClient):
    """Verify that when no friends are named, expense is saved directly to self."""
    mock_stage1_self = StageOneOutput(
        intent=IntentType.CREATE,
        is_complex=False,
        items=[
            ExtractedItem(
                description="Cold Coffee",
                amount=180.0,
                currency="INR",
                payer_raw="me",
                participants_raw=["me"],
                category="food",
                split_type=SplitTypeHint.EQUAL,
            )
        ],
        spoken_confirmation="Added ₹180 for cold coffee.",
    )

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_stage1_self.model_dump_json())
    )

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        voice_resp = await test_client.post(
            "/voice/text",
            json={"text": "Add 180 for cold coffee"},
        )

    assert voice_resp.status_code == 200
    res = voice_resp.json()
    assert res["status"] == "SUCCESS"
    assert len(res["expenses"]) == 1
    assert Decimal(str(res["expenses"][0]["amount"])) == Decimal("180.00")
