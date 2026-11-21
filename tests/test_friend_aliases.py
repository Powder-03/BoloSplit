"""Tests for custom per-user friend nicknames, aliases, conflict checks, and ranked resolution."""

import uuid
from collections.abc import AsyncGenerator
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
from app.models.user import User
from app.schemas.ai import ExtractedItem, IntentType, SplitTypeHint, StageOneOutput
from app.services.ai_pipeline import resolve_participant_ids

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DB_URL, echo=False, future=True)
async_session_maker_test = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)


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
async def multi_user_client(test_db_session: AsyncSession):
    """Client with switchable active user to test symmetric per-user aliases."""
    alice = User(
        id=uuid.uuid4(),
        email="alice@example.com",
        name="Satendra Patel",
        phone="+919999900001",
    )
    test_db_session.add(alice)
    await test_db_session.commit()
    await test_db_session.refresh(alice)

    active_user_holder = {"user": alice}

    async def override_get_session():
        yield test_db_session

    async def override_get_current_user():
        return active_user_holder["user"]

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_current_user] = override_get_current_user

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, alice, active_user_holder, test_db_session

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_custom_aliases_do_not_copy_username_and_support_symmetric_editing(multi_user_client):
    client, alice, active_holder, session = multi_user_client

    # 1. Alice adds Rishabh without custom nickname/aliases
    r1_res = await client.post(
        "/friends",
        json={
            "friend_phone": "+919800000001",
            "friend_name": "Rishabh",
        },
    )
    assert r1_res.status_code == 201
    r1_data = r1_res.json()
    assert r1_data["friend_name"] == "Rishabh"
    assert r1_data["nickname"] is None
    assert r1_data["aliases"] == []
    r1_id = uuid.UUID(r1_data["friend_id"])

    # 2. Alice adds a second friend ALSO named "Rishabh" with a distinct custom alias
    r2_res = await client.post(
        "/friends",
        json={
            "friend_phone": "+919800000002",
            "friend_name": "Rishabh",
            "nickname": "Rishabh Flat",
            "aliases": ["flatmate", "rishabh flat"],
        },
    )
    assert r2_res.status_code == 201
    r2_data = r2_res.json()
    assert r2_data["nickname"] == "Rishabh Flat"
    assert set(r2_data["aliases"]) == {"flatmate", "rishabh flat"}

    # 3. Alice updates Rishabh #1 with PATCH /friends/{friend_id}
    patch_r1 = await client.patch(
        f"/friends/{r1_id}",
        json={"nickname": "Rishabh Office", "aliases": ["office bro", "rishabh office"]},
    )
    assert patch_r1.status_code == 200
    assert patch_r1.json()["nickname"] == "Rishabh Office"
    assert set(patch_r1.json()["aliases"]) == {"office bro", "rishabh office"}

    # 4. Assigning a duplicate alias or nickname to another friend returns 409 Conflict
    dup_alias_res = await client.patch(
        f"/friends/{r1_id}",
        json={"aliases": ["flatmate"]},
    )
    assert dup_alias_res.status_code == 409

    dup_nick_res = await client.patch(
        f"/friends/{r1_id}",
        json={"nickname": "rishabh flat"},
    )
    assert dup_nick_res.status_code == 409

    # 5. Check Rishabh #1's reciprocal friendship: nickname/aliases start clean (None / [])
    r1_user = (await session.execute(select(User).where(User.id == r1_id))).scalars().first()
    assert r1_user is not None

    active_holder["user"] = r1_user
    r1_friends = (await client.get("/friends")).json()
    assert len(r1_friends) == 1
    assert r1_friends[0]["friend_id"] == str(alice.id)
    assert r1_friends[0]["friend_name"] == "Satendra Patel"
    assert r1_friends[0]["nickname"] is None
    assert r1_friends[0]["aliases"] == []

    # 6. Rishabh #1 sets his OWN custom alias for Alice (even using "office bro", valid in his own scope!)
    r1_patch_alice = await client.patch(
        f"/friends/{alice.id}",
        json={"nickname": "Satendra Bro", "aliases": ["office bro", "bhai"]},
    )
    assert r1_patch_alice.status_code == 200
    assert r1_patch_alice.json()["nickname"] == "Satendra Bro"
    assert set(r1_patch_alice.json()["aliases"]) == {"office bro", "bhai"}

    # Verify Alice's view of Rishabh #1 was NOT mutated
    active_holder["user"] = alice
    alice_friends = (await client.get("/friends")).json()
    alice_r1 = next(f for f in alice_friends if f["friend_id"] == str(r1_id))
    assert alice_r1["nickname"] == "Rishabh Office"
    assert set(alice_r1["aliases"]) == {"office bro", "rishabh office"}


@pytest.mark.asyncio
async def test_voice_add_on_friend_vs_split_with_friend_and_balances(multi_user_client):
    client, alice, _, _ = multi_user_client

    r_office = (
        await client.post(
            "/friends",
            json={
                "friend_phone": "+919811111111",
                "friend_name": "Rishabh",
                "nickname": "Rishabh Office",
                "aliases": ["rishabh office"],
            },
        )
    ).json()
    r_office_id = r_office["friend_id"]

    r_flat = (
        await client.post(
            "/friends",
            json={
                "friend_phone": "+919822222222",
                "friend_name": "Rishabh",
                "nickname": "Rishabh Flat",
                "aliases": ["rishabh flat"],
            },
        )
    ).json()
    r_flat_id = r_flat["friend_id"]

    # Case A: "add 200 on rishabh office" -> 100% owed by Rishabh Office
    mock_stage1_on = StageOneOutput(
        intent=IntentType.CREATE,
        is_complex=False,
        items=[
            ExtractedItem(
                description="Lunch",
                amount=200.0,
                payer_raw="me",
                participants_raw=["rishabh office"],
                split_type=SplitTypeHint.EQUAL,
            )
        ],
        spoken_confirmation="Added ₹200 on Rishabh Office.",
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_stage1_on.model_dump_json())
    )

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        resp_on = await client.post("/voice/text", json={"text": "add 200 on rishabh office"})

    assert resp_on.status_code == 200
    exp_on = resp_on.json()["expenses"][0]
    assert len(exp_on["splits"]) == 1
    assert exp_on["splits"][0]["user_id"] == r_office_id
    assert Decimal(str(exp_on["splits"][0]["owed_amount"])) == Decimal("200.00")

    # Case B: "add 200 split with rishabh flat" -> 50/50 split (100 each)
    mock_stage1_with = StageOneOutput(
        intent=IntentType.CREATE,
        is_complex=False,
        items=[
            ExtractedItem(
                description="Coffee",
                amount=200.0,
                payer_raw="me",
                participants_raw=["me", "rishabh flat"],
                split_type=SplitTypeHint.EQUAL,
            )
        ],
        spoken_confirmation="Added ₹200 split with Rishabh Flat.",
    )
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=mock_stage1_with.model_dump_json())
    )

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        resp_with = await client.post("/voice/text", json={"text": "add 200 split with rishabh flat"})

    assert resp_with.status_code == 200
    exp_with = resp_with.json()["expenses"][0]
    assert len(exp_with["splits"]) == 2
    flat_split = next(s for s in exp_with["splits"] if s["user_id"] == r_flat_id)
    alice_split = next(s for s in exp_with["splits"] if s["user_id"] == str(alice.id))
    assert Decimal(str(flat_split["owed_amount"])) == Decimal("100.00")
    assert Decimal(str(alice_split["owed_amount"])) == Decimal("100.00")

    # Balances surface custom nicknames for both Rishabhs
    balances = (await client.get("/balances")).json()
    by_id = {b["user_id"]: b for b in balances["balances"]}
    assert by_id[r_office_id]["nickname"] == "Rishabh Office"
    assert Decimal(str(by_id[r_office_id]["net_balance"])) == Decimal("200.00")
    assert by_id[r_flat_id]["nickname"] == "Rishabh Flat"
    assert Decimal(str(by_id[r_flat_id]["net_balance"])) == Decimal("100.00")


def test_ranked_participant_matching_avoids_substring_hijacking():
    caller_id = uuid.uuid4()
    rishabh_short_id = uuid.uuid4()
    rishabh_sharma_id = uuid.uuid4()
    vikram_id = uuid.uuid4()
    ram_id = uuid.uuid4()

    directory = [
        {
            "user_id": str(caller_id),
            "name": "Satendra",
            "nickname": "Me",
            "aliases": ["me"],
            "is_caller": True,
        },
        {
            "user_id": str(rishabh_short_id),
            "name": "Rishabh",
            "nickname": None,
            "aliases": [],
            "is_caller": False,
        },
        {
            "user_id": str(rishabh_sharma_id),
            "name": "Rishabh Sharma",
            "nickname": "Rishabh Sharma",
            "aliases": [],
            "is_caller": False,
        },
        {
            "user_id": str(vikram_id),
            "name": "Vikram",
            "nickname": None,
            "aliases": [],
            "is_caller": False,
        },
        {
            "user_id": str(ram_id),
            "name": "Ram",
            "nickname": "Ram",
            "aliases": [],
            "is_caller": False,
        },
    ]

    # "rishabh sharma" should match rishabh_sharma_id (Tier 1 exact), NOT rishabh_short_id
    assert resolve_participant_ids(["rishabh sharma"], directory, caller_id) == [rishabh_sharma_id]
    # "ram" should match ram_id, NOT vikram_id
    assert resolve_participant_ids(["ram"], directory, caller_id) == [ram_id]
    # Mixed-case "ME" still resolves caller alongside participant
    resolved_both = set(resolve_participant_ids(["ME", "ram"], directory, caller_id))
    assert resolved_both == {caller_id, ram_id}


@pytest.mark.asyncio
async def test_join_group_creates_clean_reciprocal_friendships(multi_user_client):
    client, alice, active_holder, session = multi_user_client

    grp = (await client.post("/groups", json={"name": "Office Lunch"})).json()
    invite_code = grp["invite_code"]

    bob = User(id=uuid.uuid4(), email="bob@example.com", name="Bob")
    session.add(bob)
    await session.commit()
    await session.refresh(bob)

    active_holder["user"] = bob
    join_res = await client.post(f"/groups/join/{invite_code}")
    assert join_res.status_code == 200

    # Bob sees Alice in his friends list and can immediately assign a custom nickname
    bob_friends = (await client.get("/friends")).json()
    assert any(f["friend_id"] == str(alice.id) and f["nickname"] is None for f in bob_friends)

    patch_res = await client.patch(
        f"/friends/{alice.id}",
        json={"nickname": "Satendra Lead", "aliases": ["lead"]},
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["nickname"] == "Satendra Lead"
