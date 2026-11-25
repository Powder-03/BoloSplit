"""Tests for Strictly Solo / Self Expenses endpoints and summaries."""

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
        email="solo_tester@example.com",
        name="Solo Tester",
        phone="+919876500000",
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
async def test_create_and_get_self_expenses(client_with_user):
    client, caller = client_with_user

    # 1. Initially, GET /expenses/self should be empty
    init_res = await client.get("/expenses/self")
    assert init_res.status_code == 200
    init_data = init_res.json()
    assert Decimal(str(init_data["total_spending"])) == Decimal("0.00")
    assert Decimal(str(init_data["this_month_spending"])) == Decimal("0.00")
    assert init_data["total_count"] == 0
    assert init_data["categories"] == []
    assert init_data["expenses"] == []

    # 2. Create a solo expense via POST /expenses/self
    post_res = await client.post(
        "/expenses/self",
        json={
            "description": "Petrol",
            "amount": "500.00",
            "category": "fuel",
        },
    )
    assert post_res.status_code == 201
    created_solo = post_res.json()
    solo_id = created_solo["id"]
    assert created_solo["description"] == "Petrol"
    assert Decimal(str(created_solo["amount"])) == Decimal("500.00")
    assert created_solo["category"] == "fuel"
    assert created_solo["user_id"] == str(caller.id)

    # Verify GET / PATCH / DELETE on /expenses/self/{id}
    detail_res = await client.get(f"/expenses/self/{solo_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["description"] == "Petrol"

    patch_res = await client.patch(
        f"/expenses/self/{solo_id}",
        json={"description": "Petrol Bunk", "amount": "550.00"},
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["description"] == "Petrol Bunk"
    assert Decimal(str(patch_res.json()["amount"])) == Decimal("550.00")

    # Restore amount to 500.00 for summary assertion below
    await client.patch(
        f"/expenses/self/{solo_id}",
        json={"description": "Petrol", "amount": "500.00"},
    )

    # 3. Add more solo expenses in different categories
    res2 = await client.post(
        "/expenses/self",
        json={"description": "Highway Diesel", "amount": "1500.00", "category": "fuel"},
    )
    assert res2.status_code == 201

    res3 = await client.post(
        "/expenses/self",
        json={"description": "Cold Brew Coffee", "amount": "180.00", "category": "food"},
    )
    assert res3.status_code == 201

    res4 = await client.post(
        "/expenses/self",
        json={"description": "Blinkit Groceries", "amount": "820.00", "category": "groceries"},
    )
    assert res4.status_code == 201

    # 4. Create a friend and a shared expense (should NOT appear in self expenses!)
    friend_res = await client.post(
        "/friends",
        json={"friend_name": "Rohan", "friend_phone": "+919999911111"},
    )
    assert friend_res.status_code == 201
    rohan_id = friend_res.json()["friend_id"]

    shared_res = await client.post(
        "/expenses",
        json={
            "description": "Shared Dinner with Rohan",
            "amount": "1200.00",
            "payer_id": str(caller.id),
            "splits": [
                {"user_id": str(caller.id), "owed_amount": "600.00"},
                {"user_id": str(rohan_id), "owed_amount": "600.00"},
            ],
        },
    )
    assert shared_res.status_code == 201

    # 5. Fetch GET /expenses/self
    self_res = await client.get("/expenses/self")
    assert self_res.status_code == 200
    self_data = self_res.json()

    # Total solo spend: 500 + 1500 + 180 + 820 = 3000.00 (Excluding 1200 shared dinner!)
    assert Decimal(str(self_data["total_spending"])) == Decimal("3000.00")
    assert Decimal(str(self_data["this_month_spending"])) == Decimal("3000.00")
    assert self_data["total_count"] == 4
    assert len(self_data["expenses"]) == 4

    # Verify category statistics
    cats = {c["category"]: (Decimal(str(c["total_amount"])), c["count"]) for c in self_data["categories"]}
    assert cats["fuel"] == (Decimal("2000.00"), 2)
    assert cats["groceries"] == (Decimal("820.00"), 1)
    assert cats["food"] == (Decimal("180.00"), 1)

    # 6. Test category filtering on /expenses/self
    fuel_filter_res = await client.get("/expenses/self?category=fuel")
    assert fuel_filter_res.status_code == 200
    fuel_data = fuel_filter_res.json()
    assert fuel_data["total_count"] == 2
    assert len(fuel_data["expenses"]) == 2
    assert all(e["category"] == "fuel" for e in fuel_data["expenses"])

    # 7. Verify shared /expenses does NOT include solo expenses
    scope_self_res = await client.get("/expenses?scope=self")
    assert scope_self_res.status_code == 200
    assert scope_self_res.json() == []

    scope_shared_res = await client.get("/expenses?scope=shared")
    assert scope_shared_res.status_code == 200
    assert len(scope_shared_res.json()) == 1
    assert scope_shared_res.json()[0]["description"] == "Shared Dinner with Rohan"

    scope_all_res = await client.get("/expenses")
    assert scope_all_res.status_code == 200
    assert len(scope_all_res.json()) == 1
    assert scope_all_res.json()[0]["description"] == "Shared Dinner with Rohan"

    # 8. Delete one solo expense via DELETE /expenses/self/{id}
    del_res = await client.delete(f"/expenses/self/{solo_id}")
    assert del_res.status_code == 204
    assert (await client.get(f"/expenses/self/{solo_id}")).status_code == 404
    after_del = (await client.get("/expenses/self")).json()
    assert after_del["total_count"] == 3
    assert Decimal(str(after_del["total_spending"])) == Decimal("2500.00")

    # 9. Verify /balances is NOT affected by solo expenses
    bal_res = await client.get("/balances")
    assert bal_res.status_code == 200
    bal_data = bal_res.json()
    # Rohan owes caller 600 from shared dinner; solo 2500 has 0 effect on balances
    assert Decimal(str(bal_data["total_owed_to_you"])) == Decimal("600.00")
    assert Decimal(str(bal_data["net_total"])) == Decimal("600.00")

    # 10. Test prefix route /api/v1/expenses/self
    v1_res = await client.get("/api/v1/expenses/self")
    assert v1_res.status_code == 200
    assert Decimal(str(v1_res.json()["total_spending"])) == Decimal("2500.00")


@pytest.mark.asyncio
async def test_self_expenses_category_normalization_and_empty_category(client_with_user):
    client, _ = client_with_user

    # Mix case and None/empty categories (ensures Python normalization merges cleanly without SQL GROUP BY bugs)
    r1 = await client.post("/expenses/self", json={"description": "Tea", "amount": "40.00", "category": "Food"})
    r2 = await client.post("/expenses/self", json={"description": "Snacks", "amount": "60.00", "category": "food"})
    r3 = await client.post("/expenses/self", json={"description": "Misc", "amount": "100.00", "category": None})
    assert r1.status_code == 201
    assert r2.status_code == 201
    assert r3.status_code == 201

    res = await client.get("/expenses/self")
    assert res.status_code == 200
    data = res.json()
    cats = {c["category"]: (Decimal(str(c["total_amount"])), c["count"]) for c in data["categories"]}
    assert cats["food"] == (Decimal("100.00"), 2)
    assert cats["general"] == (Decimal("100.00"), 1)

