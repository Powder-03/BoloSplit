"""Integration tests for Splitwise core features: settlements, bilateral feeds, and direct onboarding."""

import uuid
from collections.abc import AsyncGenerator
from decimal import Decimal

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from app.auth import get_current_user
from app.database import get_session
from app.main import app
from app.models.user import User

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
async def client_with_user(test_db_session: AsyncSession) -> AsyncGenerator[tuple[AsyncClient, User], None]:
    user = User(
        id=uuid.uuid4(),
        email="satendra@example.com",
        name="Satendra Patel",
        phone="+919999988888",
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
        yield c, user

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_reciprocal_friend_and_bilateral_ledger(client_with_user):
    client, caller = client_with_user

    # Add friend by phone
    friend_res = await client.post(
        "/friends",
        json={
            "friend_phone": "+919876543210",
            "friend_name": "Rohan Verma",
            "nickname": "Rohan",
        },
    )
    assert friend_res.status_code == 201
    friend_data = friend_res.json()
    rohan_id = uuid.UUID(friend_data["friend_id"])

    # Create an expense paid by caller split equally with Rohan
    exp_res = await client.post(
        "/expenses",
        json={
            "description": "Zomato Biryani",
            "amount": "600.00",
            "payer_id": str(caller.id),
            "splits": [
                {"user_id": str(caller.id), "owed_amount": "300.00"},
                {"user_id": str(rohan_id), "owed_amount": "300.00"},
            ],
        },
    )
    assert exp_res.status_code == 201

    # Check Bilateral Activity Feed
    feed_res = await client.get(f"/friends/{rohan_id}/expenses")
    assert feed_res.status_code == 200
    feed = feed_res.json()
    assert feed["friend_name"] == "Rohan Verma"
    assert Decimal(str(feed["net_balance"])) == Decimal("300.00")
    assert "Rohan Verma owes you ₹300.00" in feed["summary_text"]
    assert len(feed["timeline"]) == 1
    assert feed["timeline"][0]["type"] == "EXPENSE"

    # Rohan settles up: Rohan pays caller 300 via UPI
    # To test settlement recording:
    settle_res = await client.post(
        "/settlements",
        json={
            "payee_id": str(caller.id),
            "amount": "100.00",
        },
    )
    # Cannot settle with yourself
    assert settle_res.status_code == 400

    # Caller records that Rohan settled up (or caller records paying Rohan):
    # Here caller records paying Rohan 300:
    pay_res = await client.post(
        "/settlements",
        json={
            "payee_id": str(rohan_id),
            "amount": "300.00",
            "payment_method": "UPI",
            "notes": "Paid Rohan back",
        },
    )
    assert pay_res.status_code == 201

    # Check feed again: net balance increased by 300
    feed_after = (await client.get(f"/friends/{rohan_id}/expenses")).json()
    assert Decimal(str(feed_after["net_balance"])) == Decimal("600.00")
    assert len(feed_after["timeline"]) == 2


@pytest.mark.asyncio
async def test_direct_group_member_onboarding(client_with_user):
    client, caller = client_with_user

    # Create Group
    grp_res = await client.post("/groups", json={"name": "Flat 402"})
    assert grp_res.status_code == 201
    group_id = grp_res.json()["id"]

    # Directly add member by email
    add_email = await client.post(
        f"/groups/{group_id}/members",
        json={"identifier": "roommate1@flat.com", "name": "Roomie One", "role": "member"},
    )
    assert add_email.status_code == 201
    assert add_email.json()["name"] == "Roomie One"

    # Directly add member by phone
    add_phone = await client.post(
        f"/groups/{group_id}/members",
        json={"identifier": "+918888877777", "name": "Roomie Two", "role": "member"},
    )
    assert add_phone.status_code == 201
    assert add_phone.json()["phone"] == "+918888877777"

    # Get Group Details with roster
    grp_detail = (await client.get(f"/groups/{group_id}")).json()
    assert grp_detail["name"] == "Flat 402"
    assert len(grp_detail["members"]) == 3  # Caller + Roomie One + Roomie Two

    # Post group expense
    exp = await client.post(
        "/expenses",
        json={
            "group_id": group_id,
            "description": "WiFi Bill",
            "amount": "999.00",
            "payer_id": str(caller.id),
        },
    )
    assert exp.status_code == 201

    # Check group feed
    grp_feed = (await client.get(f"/groups/{group_id}/expenses")).json()
    assert len(grp_feed["timeline"]) == 1
    assert Decimal(str(grp_feed["total_spending"])) == Decimal("999.00")
