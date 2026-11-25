"""Tests for unified audit activity logging (`GET /activity`) across manual and voice workflows."""

import uuid
from collections.abc import AsyncGenerator
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import app.database as db_mod
from app.auth import get_current_user
from app.database import get_session
from app.main import app
from app.models.activity import ActivityLog
from app.models.expense import Expense, ExpenseSplit
from app.models.settlement import Settlement
from app.models.user import User
from app.schemas.ai import IntentType, StageOneOutput
from app.services.activity_service import backfill_missing_activity_logs

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(TEST_DB_URL, echo=False, future=True)
async_session_maker_test = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)


@pytest.fixture(scope="session", autouse=True)
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with async_session_maker_test() as session:
        yield session
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


@pytest.fixture
async def multi_user_client(db_session: AsyncSession):
    """Test client allowing switching the active authenticated user."""
    alice = User(id=uuid.uuid4(), email="alice@example.com", name="Alice")
    bob = User(id=uuid.uuid4(), email="bob@example.com", name="Bob")
    stranger = User(id=uuid.uuid4(), email="stranger@example.com", name="Stranger")
    db_session.add_all([alice, bob, stranger])
    await db_session.commit()

    active_user_holder = {"user": alice}

    async def override_get_session():
        yield db_session

    async def override_get_current_user():
        return active_user_holder["user"]

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_current_user] = override_get_current_user

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, active_user_holder, alice, bob, stranger, db_session

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_manual_expense_lifecycle_logs_create_update_and_delete(multi_user_client):
    client, active_holder, alice, bob, stranger, _ = multi_user_client

    # Alice creates a shared group with Bob
    grp_res = await client.post("/groups", json={"name": "Goa Trip"})
    assert grp_res.status_code == 201
    group_id = grp_res.json()["id"]

    add_res = await client.post(
        f"/groups/{group_id}/members",
        json={"identifier": bob.email, "name": "Bob"},
    )
    assert add_res.status_code == 201

    # 1. Alice creates a shared expense
    exp_res = await client.post(
        "/expenses",
        json={
            "description": "Beach Dinner",
            "amount": "1000.00",
            "payer_id": str(alice.id),
            "group_id": group_id,
            "category": "food",
        },
    )
    assert exp_res.status_code == 201
    expense_id = exp_res.json()["id"]

    # 2. Alice updates the expense
    patch_res = await client.patch(
        f"/expenses/{expense_id}",
        json={"description": "Beach Dinner & Drinks", "amount": "1200.00"},
    )
    assert patch_res.status_code == 200

    # 3. Bob deletes the expense
    active_holder["user"] = bob
    del_res = await client.delete(f"/expenses/{expense_id}")
    assert del_res.status_code == 204

    # Balances & active expense list should both be clean after delete
    bal_res = await client.get("/balances")
    assert bal_res.status_code == 200
    assert Decimal(str(bal_res.json()["net_total"])) == Decimal("0.00")

    active_expenses = await client.get("/expenses")
    assert active_expenses.status_code == 200
    assert active_expenses.json() == []

    # 4. Alice fetches GET /activity -> sees DELETE, UPDATE, CREATE in reverse chronological order
    active_holder["user"] = alice
    alice_act = await client.get("/activity")
    assert alice_act.status_code == 200
    items = alice_act.json()
    assert [i["event_type"] for i in items] == [
        "EXPENSE_DELETED",
        "EXPENSE_UPDATED",
        "EXPENSE_CREATED",
    ]

    deleted_item = items[0]
    assert deleted_item["entity_type"] == "EXPENSE"
    assert deleted_item["entity_id"] == expense_id
    assert deleted_item["actor_id"] == str(bob.id)
    assert deleted_item["actor_name"] == "Bob"
    assert deleted_item["is_actor_you"] is False
    assert deleted_item["group_name"] == "Goa Trip"
    assert deleted_item["is_entity_deleted"] is True
    assert "Bob deleted 'Beach Dinner & Drinks'" in deleted_item["summary_text"]

    # Earlier CREATE / UPDATE entries also flag is_entity_deleted=True so clients don't open dead details
    assert items[1]["is_entity_deleted"] is True
    assert items[2]["is_entity_deleted"] is True
    assert "You added 'Beach Dinner'" in items[2]["summary_text"]

    # 5. Bob fetches GET /activity -> sees "You deleted..."
    active_holder["user"] = bob
    bob_act = (await client.get("/activity")).json()
    assert bob_act[0]["is_actor_you"] is True
    assert "You deleted 'Beach Dinner & Drinks'" in bob_act[0]["summary_text"]

    # 6. Unrelated user Stranger sees nothing
    active_holder["user"] = stranger
    stranger_act = (await client.get("/activity")).json()
    assert stranger_act == []


@pytest.mark.asyncio
async def test_voice_delete_logs_activity_and_solo_expenses_are_excluded(multi_user_client):
    client, active_holder, alice, bob, _, _ = multi_user_client
    active_holder["user"] = alice

    # Create friend relationship with Bob and a shared expense
    await client.post("/friends", json={"friend_email": bob.email, "friend_name": "Bob"})
    exp_res = await client.post(
        "/expenses",
        json={
            "description": "Movie Tickets",
            "amount": "600.00",
            "payer_id": str(alice.id),
            "splits": [
                {"user_id": str(alice.id), "owed_amount": "300.00"},
                {"user_id": str(bob.id), "owed_amount": "300.00"},
            ],
        },
    )
    assert exp_res.status_code == 201
    expense_id = exp_res.json()["id"]

    # Solo expense create + delete should NOT pollute shared /activity
    solo_res = await client.post(
        "/expenses/self",
        json={"description": "Petrol", "amount": "500.00", "category": "fuel"},
    )
    assert solo_res.status_code == 201
    solo_id = solo_res.json()["id"]
    assert (await client.delete(f"/expenses/self/{solo_id}")).status_code == 204

    # Voice DELETE of shared Movie Tickets
    stage1_del = StageOneOutput(
        intent=IntentType.DELETE,
        is_complex=False,
        target_record_id=expense_id,
        spoken_confirmation="Done! Deleted Movie Tickets.",
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=MagicMock(text=stage1_del.model_dump_json())
    )

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_client):
        voice_resp = await client.post("/voice/text", json={"text": "Delete movie tickets"})
        assert voice_resp.status_code == 200
        assert voice_resp.json()["action_type"] == "DELETE"

    act_items = (await client.get("/activity")).json()
    assert [i["event_type"] for i in act_items] == ["EXPENSE_DELETED", "EXPENSE_CREATED"]
    assert act_items[0]["metadata"].get("source") == "voice"
    assert all(i["description"] != "Petrol" for i in act_items)


@pytest.mark.asyncio
async def test_settlement_lifecycle_logs_and_filters(multi_user_client):
    client, active_holder, alice, bob, _, _ = multi_user_client
    active_holder["user"] = alice

    # Create and then undo/delete a settlement
    stl_res = await client.post(
        "/settlements",
        json={"payee_id": str(bob.id), "amount": "250.00", "payment_method": "UPI"},
    )
    assert stl_res.status_code == 201
    stl_id = stl_res.json()["id"]

    del_stl = await client.delete(f"/settlements/{stl_id}")
    assert del_stl.status_code == 204

    all_items = (await client.get("/activity")).json()
    assert [i["event_type"] for i in all_items] == ["SETTLEMENT_DELETED", "SETTLEMENT_CREATED"]
    assert all_items[0]["entity_type"] == "SETTLEMENT"
    assert all_items[0]["is_entity_deleted"] is True

    # Filter by event_type
    deleted_only = (await client.get("/activity", params={"event_type": "SETTLEMENT_DELETED"})).json()
    assert len(deleted_only) == 1
    assert deleted_only[0]["id"] == all_items[0]["id"]


@pytest.mark.asyncio
async def test_backfill_missing_activity_logs_is_idempotent(db_session: AsyncSession):
    alice = User(id=uuid.uuid4(), email="a1@example.com", name="Alice")
    bob = User(id=uuid.uuid4(), email="b1@example.com", name="Bob")
    db_session.add_all([alice, bob])
    await db_session.flush()

    exp = Expense(
        description="Legacy Cab",
        amount=Decimal("400.00"),
        payer_id=alice.id,
        is_personal=False,
        is_deleted=False,
    )
    db_session.add(exp)
    await db_session.flush()
    db_session.add_all(
        [
            ExpenseSplit(expense_id=exp.id, user_id=alice.id, owed_amount=Decimal("200.00")),
            ExpenseSplit(expense_id=exp.id, user_id=bob.id, owed_amount=Decimal("200.00")),
        ]
    )

    stl = Settlement(
        payer_id=bob.id,
        payee_id=alice.id,
        amount=Decimal("200.00"),
        is_deleted=False,
    )
    db_session.add(stl)
    await db_session.commit()

    first_count = await backfill_missing_activity_logs(db_session)
    await db_session.commit()
    assert first_count == 2

    second_count = await backfill_missing_activity_logs(db_session)
    await db_session.commit()
    assert second_count == 0


@pytest.mark.asyncio
async def test_init_db_upgrades_legacy_tables_and_backfills_idempotently(tmp_path, monkeypatch):
    legacy_db = tmp_path / "legacy_upgrade.db"
    legacy_engine = create_async_engine(f"sqlite+aiosqlite:///{legacy_db}", echo=False, future=True)
    legacy_maker = async_sessionmaker(legacy_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)

    uid = uuid.uuid4().hex
    eid = uuid.uuid4().hex
    se_id = uuid.uuid4().hex

    # Simulate legacy schema before `is_personal` and `activity_logs` existed
    async with legacy_engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE TABLE users (id CHAR(32) PRIMARY KEY, firebase_uid VARCHAR(128), email VARCHAR(255) UNIQUE NOT NULL, name VARCHAR(255) NOT NULL, phone VARCHAR(50), created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)"
            )
        )
        await conn.execute(
            text(
                "CREATE TABLE groups (id CHAR(32) PRIMARY KEY, name VARCHAR(255) NOT NULL, invite_code VARCHAR(20) UNIQUE NOT NULL, created_by CHAR(32) NOT NULL, created_at DATETIME NOT NULL)"
            )
        )
        await conn.execute(
            text(
                "CREATE TABLE expenses (id CHAR(32) PRIMARY KEY, group_id CHAR(32), description VARCHAR(255) NOT NULL, amount NUMERIC(12, 2) NOT NULL, currency VARCHAR(10) NOT NULL, payer_id CHAR(32) NOT NULL, category VARCHAR(100), expense_date DATETIME NOT NULL, is_deleted BOOLEAN NOT NULL, created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)"
            )
        )
        await conn.execute(
            text(
                "CREATE TABLE self_expenses (id CHAR(32) PRIMARY KEY, user_id CHAR(32) NOT NULL, description VARCHAR(255) NOT NULL, amount NUMERIC(12, 2) NOT NULL, currency VARCHAR(10) NOT NULL, category VARCHAR(100), expense_date DATETIME NOT NULL, is_deleted BOOLEAN NOT NULL, created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO users VALUES (:id, NULL, 'u@example.com', 'User', NULL, '2026-11-20 10:00:00', '2026-11-20 10:00:00')"
            ),
            {"id": uid},
        )
        await conn.execute(
            text(
                "INSERT INTO expenses VALUES (:id, NULL, 'Shared Lunch', 300.00, 'INR', :uid, 'food', '2026-11-20 10:00:00', 0, '2026-11-20 10:00:00', '2026-11-20 10:00:00')"
            ),
            {"id": eid, "uid": uid},
        )
        await conn.execute(
            text(
                "INSERT INTO self_expenses VALUES (:id, :uid, 'Solo Chai', 30.00, 'INR', 'food', '2026-11-20 10:00:00', 0, '2026-11-20 10:00:00', '2026-11-20 10:00:00')"
            ),
            {"id": se_id, "uid": uid},
        )

    monkeypatch.setattr(db_mod, "engine", legacy_engine)
    monkeypatch.setattr(db_mod, "async_session_maker", legacy_maker)

    # First run upgrades schema, migrates self_expenses, creates activity_logs, and backfills
    await db_mod.init_db()
    # Second run is completely idempotent
    await db_mod.init_db()

    async with legacy_maker() as session:
        cols = {
            row[1]
            for row in (await session.execute(text("PRAGMA table_info(expenses);"))).all()
        }
        assert "is_personal" in cols

        solo_row = (
            await session.execute(text("SELECT is_personal FROM expenses WHERE id = :id"), {"id": se_id})
        ).first()
        assert solo_row is not None and bool(solo_row[0]) is True

        from sqlmodel import select as sm_select

        logs = (await session.execute(sm_select(ActivityLog))).scalars().all()
        assert len(logs) == 1
        assert logs[0].description == "Shared Lunch"
        assert logs[0].event_type == "EXPENSE_CREATED"

    await legacy_engine.dispose()
