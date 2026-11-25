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
    ExactSplitItem,
    ExtractedItem,
    IntentType,
    PercentageSplitItem,
    SettleExtraction,
    ShareSplitItem,
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
    client, caller, session = client_with_user

    seed = await client.post(
        "/expenses/self",
        json={"description": "Dinner", "amount": "400.00", "category": "food"},
    )
    assert seed.status_code == 201
    seed_id = seed.json()["id"]

    mock_s1 = StageOneOutput(
        intent=IntentType.UPDATE,
        confidence="HIGH",
        is_complex=True,
        items=[ExtractedItem(description="Dinner", amount=450.0, scope="SELF", participants_raw=["me"])],
    )
    mock_s2 = StageTwoResolution(
        action_type="UPDATE",
        target_record_id=seed_id,
        target_source="self",
        confidence="MEDIUM",
        description="Dinner",
        amount=Decimal("450.00"),
        clarification_question="Not sure which dinner expense you meant. Please speak again.",
    )
    mock_client = MagicMock()

    async def mock_gen(model, **kwargs):
        if "flash" in model:
            return MagicMock(text=mock_s1.model_dump_json())
        return MagicMock(text=mock_s2.model_dump_json())

    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_gen)

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        resp = await client.post("/voice/text", json={"text": "Update dinner to 450"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "ERROR"
    assert resp.json()["error_code"] == "LOW_CONFIDENCE"
    assert resp.json()["count"] == 0

    # Seeded self expense remained untouched
    check = (await client.get(f"/expenses/self/{seed_id}")).json()
    assert Decimal(str(check["amount"])) == Decimal("400.00")


@pytest.mark.asyncio
async def test_mixed_multi_create_separates_self_and_shared_expenses(client_with_user):
    client, caller, session = client_with_user

    f_res = (
        await client.post(
            "/friends",
            json={"friend_phone": "+919877770011", "friend_name": "Rishabh", "nickname": "Rishabh"},
        )
    ).json()
    rishabh_id = f_res["friend_id"]

    mock_multi = StageOneOutput(
        intent=IntentType.MULTI_CREATE,
        confidence="HIGH",
        is_complex=False,
        items=[
            ExtractedItem(
                description="Coffee",
                amount=120.0,
                scope="SELF",
                participants_raw=["me"],
                category="food",
            ),
            ExtractedItem(
                description="Lunch",
                amount=500.0,
                scope="SHARED",
                participants_raw=["rishabh"],
                category="food",
            ),
        ],
        spoken_confirmation="Added ₹120 solo coffee and ₹500 on Rishabh.",
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_multi.model_dump_json())
    )

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        resp = await client.post("/voice/text", json={"text": "120 coffee and 500 on Rishabh"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["count"] == 2
    assert len(data["self_expenses"]) == 1
    assert len(data["expenses"]) == 1
    assert data["self_expenses"][0]["description"] == "Coffee"
    assert data["expenses"][0]["description"] == "Lunch"
    assert data["expenses"][0]["splits"][0]["user_id"] == rishabh_id

    self_summary = (await client.get("/expenses/self")).json()
    assert self_summary["total_count"] == 1
    assert Decimal(str(self_summary["total_spending"])) == Decimal("120.00")

    shared_list = (await client.get("/expenses")).json()
    assert len(shared_list) == 1
    assert shared_list[0]["description"] == "Lunch"


@pytest.mark.asyncio
async def test_resolve_expense_date_anchors_today_and_past_dates():
    from datetime import UTC, datetime, timedelta

    from app.services.ai_pipeline import IST, resolve_expense_date

    today_ist = datetime.now(IST).date()
    resolved_today = resolve_expense_date(today_ist.isoformat())
    assert resolved_today.astimezone(IST).date() == today_ist
    # Must preserve real current time instead of pinning to 00:00 UTC (5:30 AM IST)
    assert abs((datetime.now(UTC) - resolved_today).total_seconds()) < 10

    past_date = today_ist - timedelta(days=2)
    resolved_past = resolve_expense_date(past_date.isoformat())
    assert resolved_past.astimezone(IST).date() == past_date
    assert resolved_past.astimezone(IST).hour == 12
    assert resolved_past.astimezone(IST).minute == 0


@pytest.mark.asyncio
async def test_on_friend_expense_excluded_from_self_and_isolated_across_users(test_db_session):
    user_a = User(id=uuid.uuid4(), email="usera@example.com", name="User A", phone="+919000010001")
    user_c = User(id=uuid.uuid4(), email="userc@example.com", name="User C", phone="+919000010003")
    test_db_session.add(user_a)
    test_db_session.add(user_c)
    await test_db_session.commit()
    await test_db_session.refresh(user_a)
    await test_db_session.refresh(user_c)

    active_user = {"user": user_a}

    async def override_get_session():
        yield test_db_session

    async def override_get_current_user():
        return active_user["user"]

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_current_user] = override_get_current_user

    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            f_res = await c.post(
                "/friends",
                json={"friend_phone": "+919800000099", "friend_name": "Rishabh", "nickname": "Rishabh"},
            )
            assert f_res.status_code == 201

            mock_s1 = StageOneOutput(
                intent=IntentType.CREATE,
                confidence="HIGH",
                is_complex=False,
                items=[
                    ExtractedItem(
                        description="Dinner",
                        amount=500.0,
                        payer_raw="me",
                        participants_raw=["rishabh"],
                        split_type=SplitTypeHint.EQUAL,
                    )
                ],
                spoken_confirmation="Added ₹500 on Rishabh.",
            )
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(
                return_value=MagicMock(text=mock_s1.model_dump_json())
            )
            with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
                voice_res = await c.post("/voice/text", json={"text": "add 500 rupees on rishabh"})
            assert voice_res.status_code == 200
            exp_id = voice_res.json()["expenses"][0]["id"]

            # User A sees it in /expenses & /expenses?scope=shared, NOT in /expenses/self or scope=self
            all_a = (await c.get("/expenses")).json()
            assert [e["id"] for e in all_a] == [exp_id]
            shared_a = (await c.get("/expenses?scope=shared")).json()
            assert [e["id"] for e in shared_a] == [exp_id]
            self_summary_a = (await c.get("/expenses/self")).json()
            assert self_summary_a["total_count"] == 0
            assert (await c.get("/expenses?scope=self")).json() == []

            # Unrelated User C must NOT see or mutate User A's expense
            active_user["user"] = user_c
            assert (await c.get("/expenses")).json() == []
            assert (await c.get(f"/expenses/{exp_id}")).status_code == 403
            assert (await c.patch(f"/expenses/{exp_id}", json={"amount": 100})).status_code == 403
            assert (await c.delete(f"/expenses/{exp_id}")).status_code == 403
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_friend_payer_exact_shares_group_default_and_incoming_settle(client_with_user):
    """Verify all newly fixed pipeline capabilities:
    1. Friend as payer ('Sahaj paid 500 for dinner with me')
    2. EXACT and SHARES splits
    3. Group expense default split across all members
    4. Preserving PERCENTAGE splits on PATCH amount update
    5. Incoming voice settlement ('Sahaj paid me 200')
    """
    client, caller, _ = client_with_user

    f_res = (
        await client.post(
            "/friends",
            json={"friend_phone": "+919877771122", "friend_name": "Sahaj", "nickname": "Sahaj"},
        )
    ).json()
    sahaj_id = f_res["friend_id"]

    # 1. Friend paid 500 split equally with me -> Sahaj is payer, caller owes Sahaj 250
    mock_friend_paid = StageOneOutput(
        intent=IntentType.CREATE,
        confidence="HIGH",
        items=[
            ExtractedItem(
                description="Dinner",
                amount=500.0,
                scope="SHARED",
                payer_raw="Sahaj",
                participants_raw=["me", "Sahaj"],
                split_type=SplitTypeHint.EQUAL,
            )
        ],
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_friend_paid.model_dump_json())
    )
    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        r1 = await client.post("/voice/text", json={"text": "Sahaj paid 500 for dinner with me"})
    assert r1.status_code == 200
    exp1 = r1.json()["expenses"][0]
    assert exp1["payer_id"] == sahaj_id
    assert len(exp1["splits"]) == 2

    # 2. EXACT split: 500 total, 350 for Sahaj, 150 for me
    mock_exact = StageOneOutput(
        intent=IntentType.CREATE,
        confidence="HIGH",
        items=[
            ExtractedItem(
                description="Lunch",
                amount=500.0,
                scope="SHARED",
                payer_raw="me",
                split_type=SplitTypeHint.EXACT,
                exact_splits=[
                    ExactSplitItem(participant_raw="Sahaj", amount=350.0),
                    ExactSplitItem(participant_raw="me", amount=150.0),
                ],
            )
        ],
    )
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_exact.model_dump_json())
    )
    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        r2 = await client.post("/voice/text", json={"text": "500 lunch, 350 Sahaj and 150 me"})
    exp2 = r2.json()["expenses"][0]
    splits2 = {s["user_id"]: Decimal(str(s["owed_amount"])) for s in exp2["splits"]}
    assert splits2[sahaj_id] == Decimal("350.00")
    assert splits2[str(caller.id)] == Decimal("150.00")

    # 3. SHARES split (2:1) & PERCENTAGE split preserved on PATCH amount change
    mock_pct = StageOneOutput(
        intent=IntentType.CREATE,
        confidence="HIGH",
        items=[
            ExtractedItem(
                description="Cab",
                amount=500.0,
                scope="SHARED",
                payer_raw="me",
                split_type=SplitTypeHint.PERCENTAGE,
                percentage_splits=[
                    PercentageSplitItem(participant_raw="me", percentage=20.0),
                    PercentageSplitItem(participant_raw="Sahaj", percentage=80.0),
                ],
            )
        ],
    )
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_pct.model_dump_json())
    )
    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        r3 = await client.post("/voice/text", json={"text": "500 cab 20% me 80% Sahaj"})
    cab_id = r3.json()["expenses"][0]["id"]

    # Update cab amount to 1000 via PATCH -> must keep 20%/80% (200 / 800), NOT 500/500!
    patched = (await client.patch(f"/expenses/{cab_id}", json={"amount": "1000.00"})).json()
    patched_splits = {s["user_id"]: Decimal(str(s["owed_amount"])) for s in patched["splits"]}
    assert patched_splits[str(caller.id)] == Decimal("200.00")
    assert patched_splits[sahaj_id] == Decimal("800.00")

    # 4. Group expense without naming individual members splits across all group members
    grp = (await client.post("/groups", json={"name": "Coorg Trip"})).json()
    gid = grp["id"]
    await client.post(f"/groups/{gid}/members", json={"identifier": sahaj_id})

    mock_grp_exp = StageOneOutput(
        intent=IntentType.CREATE,
        confidence="HIGH",
        items=[
            ExtractedItem(
                description="Snacks",
                amount=300.0,
                scope="SHARED",
                payer_raw="me",
                group_raw="Coorg Trip",
                participants_raw=["me"],
                split_type=SplitTypeHint.EQUAL,
            )
        ],
    )
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_grp_exp.model_dump_json())
    )
    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        r4 = await client.post("/voice/text", json={"text": "Add 300 snacks in Coorg Trip"})
    exp4 = r4.json()["expenses"][0]
    assert len(exp4["splits"]) == 2
    assert all(Decimal(str(s["owed_amount"])) == Decimal("150.00") for s in exp4["splits"])

    # 5. Incoming settlement: "Sahaj paid me 200" -> payer_id == sahaj_id, payee_id == caller.id
    mock_incoming_settle = StageOneOutput(
        intent=IntentType.SETTLE,
        confidence="HIGH",
        settle_item=SettleExtraction(payer_raw="Sahaj", payee_raw="me", amount=200.0),
    )
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_incoming_settle.model_dump_json())
    )
    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        r5 = await client.post("/voice/text", json={"text": "Sahaj paid me 200"})
    assert r5.status_code == 200
    s_data = r5.json()["settlement"]
    assert s_data["payer_id"] == sahaj_id
    assert s_data["payee_id"] == str(caller.id)

    # 6. Second-Pass Fix: "Sahaj paid 500 for dinner" where LLM sets participants_raw=["Sahaj"]
    # still includes caller.id in splits so caller isn't locked out of the expense
    mock_friend_only_part = StageOneOutput(
        intent=IntentType.CREATE,
        confidence="HIGH",
        items=[
            ExtractedItem(
                description="Late Dinner",
                amount=400.0,
                scope="SHARED",
                payer_raw="Sahaj",
                participants_raw=["Sahaj"],
                split_type=SplitTypeHint.EQUAL,
            )
        ],
    )
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_friend_only_part.model_dump_json())
    )
    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        r6 = await client.post("/voice/text", json={"text": "Sahaj paid 400 for late dinner"})
    exp6 = r6.json()["expenses"][0]
    assert exp6["payer_id"] == sahaj_id
    assert {s["user_id"] for s in exp6["splits"]} == {sahaj_id, str(caller.id)}

    # 7. Second-Pass Fix: Single-pass UPDATE preserves specific description ("Late Dinner") & category
    # when user says "Update dinner to 450"
    mock_single_pass_update = StageOneOutput(
        intent=IntentType.UPDATE,
        confidence="HIGH",
        target_record_id=exp6["id"],
        items=[
            ExtractedItem(
                description="Dinner",
                amount=450.0,
                category="general",
            )
        ],
    )
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_single_pass_update.model_dump_json())
    )
    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        r7 = await client.post("/voice/text", json={"text": "Update dinner to 450"})
    assert r7.status_code == 200
    exp7 = r7.json()["expenses"][0]
    assert Decimal(str(exp7["amount"])) == Decimal("450.00")
    assert exp7["description"] == "Late Dinner"  # Not overwritten by shorter search keyword "Dinner"
    assert exp7["payer_name"] == "Sahaj"
