"""Tests verifying candidate ambiguity handling when multiple records match."""

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
from app.models.expense import Expense
from app.models.user import User
from app.schemas.ai import (
    CandidateSearchQuery,
    IntentType,
    StageOneOutput,
    StageTwoResolution,
)

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DB_URL, echo=False, future=True)
async_session_maker_test = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)


@pytest.fixture
async def test_session():
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with async_session_maker_test() as session:
        yield session
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


@pytest.fixture
async def client_with_two_candidates(test_session: AsyncSession):
    user = User(id=uuid.uuid4(), email="tester@example.com", name="Tester")
    test_session.add(user)
    await test_session.commit()
    await test_session.refresh(user)

    # Seed 2 conflicting petrol expenses
    exp1 = Expense(
        id=uuid.uuid4(),
        description="Petrol bunk HP",
        amount=Decimal("500.00"),
        payer_id=user.id,
        category="fuel",
    )
    exp2 = Expense(
        id=uuid.uuid4(),
        description="Petrol bunk IndianOil",
        amount=Decimal("800.00"),
        payer_id=user.id,
        category="fuel",
    )
    test_session.add_all([exp1, exp2])
    await test_session.commit()

    async def override_get_session():
        yield test_session

    async def override_get_current_user():
        return user

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_current_user] = override_get_current_user

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, exp1, exp2

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ambiguous_candidate_conflict_resolution(client_with_two_candidates):
    client, exp1, exp2 = client_with_two_candidates

    mock_stage1 = StageOneOutput(
        intent=IntentType.UPDATE,
        is_complex=True,
        search_query=CandidateSearchQuery(description_keywords=["petrol"]),
    )

    # Stage 2 flags ambiguity between exp1 and exp2
    mock_stage2 = StageTwoResolution(
        action_type="UPDATE",
        confidence="AMBIGUOUS",
        ambiguous_candidate_ids=[str(exp1.id), str(exp2.id)],
        description="Petrol",
        amount=Decimal("600.00"),
        clarification_question="Found two petrol expenses (₹500 and ₹800). Which one would you like to update?",
        spoken_confirmation="Aapke do petrol bills hain, kaunsa update karna hai?",
    )

    mock_genai_client = MagicMock()

    async def mock_gen(model, **kwargs):
        res = MagicMock()
        if "flash" in model:
            res.text = mock_stage1.model_dump_json()
        else:
            res.text = mock_stage2.model_dump_json()
        return res

    mock_genai_client.aio.models.generate_content = AsyncMock(side_effect=mock_gen)

    with patch("app.services.ai_pipeline.get_genai_client", return_value=mock_genai_client):
        resp = await client.post("/voice/text", json={"text": "Update yesterday's petrol to 600"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "AMBIGUOUS"
    assert data["is_ambiguous"] is True
    assert "Found two petrol expenses" in data["clarification_prompt"]
