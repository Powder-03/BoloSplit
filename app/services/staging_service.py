"""Staging service: queue stacking, duplicate check, and atomic promotion to ledger."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.expense import Expense, ExpenseSplit
from app.models.staging import StagedExpense
from app.schemas.staging import BatchApproveResponse
from app.services.ledger_service import calculate_equal_splits


async def check_is_duplicate(
    session: AsyncSession,
    user_id: uuid.UUID,
    amount: Decimal,
    category: str | None,
    window_minutes: int = 10,
) -> bool:
    """Check if an identical amount + category exists in PENDING staged items in last 10 minutes."""
    cutoff = datetime.now(UTC) - timedelta(minutes=window_minutes)
    query = (
        select(StagedExpense)
        .where(
            StagedExpense.user_id == user_id,
            StagedExpense.status == "PENDING",
            StagedExpense.amount == amount,
            StagedExpense.created_at >= cutoff,
        )
    )
    if category:
        query = query.where(StagedExpense.category == category)

    result = await session.execute(query)
    existing = result.scalars().first()
    return existing is not None


async def create_staged_expense(
    session: AsyncSession,
    user_id: uuid.UUID,
    description: str,
    amount: Decimal,
    action_type: str = "CREATE",
    group_id: uuid.UUID | None = None,
    target_expense_id: uuid.UUID | None = None,
    payer_id: uuid.UUID | None = None,
    payer_name: str | None = None,
    split_data: list[dict[str, Any]] | None = None,
    expense_date: datetime | None = None,
    category: str | None = "general",
    confidence: str = "HIGH",
    candidate_options: list[dict[str, Any]] | None = None,
    raw_voice_transcript: str | None = None,
    spoken_confirmation: str | None = None,
) -> StagedExpense:
    """Create a staged expense review card with duplicate detection."""
    is_dup = False
    if action_type == "CREATE":
        is_dup = await check_is_duplicate(
            session=session,
            user_id=user_id,
            amount=amount,
            category=category,
        )

    staged = StagedExpense(
        user_id=user_id,
        group_id=group_id,
        action_type=action_type,
        status="PENDING",
        target_expense_id=target_expense_id,
        description=description,
        amount=amount,
        currency="INR",
        payer_id=payer_id,
        payer_name=payer_name,
        split_data=split_data or [],
        expense_date=expense_date or datetime.now(UTC),
        category=category or "general",
        confidence=confidence,
        candidate_options=candidate_options,
        is_possible_duplicate=is_dup,
        raw_voice_transcript=raw_voice_transcript,
        spoken_confirmation=spoken_confirmation,
    )
    session.add(staged)
    await session.commit()
    await session.refresh(staged)
    return staged


async def get_staged_expenses(
    session: AsyncSession,
    user_id: uuid.UUID,
    status: str = "PENDING",
    group_id: uuid.UUID | None = None,
) -> list[StagedExpense]:
    """Retrieve staged items for a user/group ordered by creation time desc."""
    query = (
        select(StagedExpense)
        .where(
            StagedExpense.user_id == user_id,
            StagedExpense.status == status,
        )
        .order_by(StagedExpense.created_at.desc())
    )
    if group_id:
        query = query.where(StagedExpense.group_id == group_id)

    result = await session.execute(query)
    return list(result.scalars().all())


async def update_staged_expense(
    session: AsyncSession,
    staged_id: uuid.UUID,
    user_id: uuid.UUID,
    update_data: dict[str, Any],
) -> StagedExpense | None:
    """Patch user-edited fields on an unapproved staged review card."""
    query = select(StagedExpense).where(
        StagedExpense.id == staged_id,
        StagedExpense.user_id == user_id,
        StagedExpense.status == "PENDING",
    )
    result = await session.execute(query)
    staged = result.scalars().first()
    if not staged:
        return None

    for key, value in update_data.items():
        if value is not None and hasattr(staged, key):
            setattr(staged, key, value)

    staged.updated_at = datetime.now(UTC)
    session.add(staged)
    await session.commit()
    await session.refresh(staged)
    return staged


async def approve_staged_item(
    session: AsyncSession,
    staged: StagedExpense,
) -> uuid.UUID | None:
    """
    Atomically promote a single staged item to the main ledger.

    - CREATE: Inserts Expense + ExpenseSplit rows and marks Staged.status = APPROVED.
    - UPDATE: Updates target Expense and marks Staged.status = APPROVED.
    - DELETE: Marks target Expense.is_deleted = True and marks Staged.status = APPROVED.
    """
    approved_expense_id: uuid.UUID | None = None

    if staged.action_type == "CREATE":
        # Determine payer
        payer_id = staged.payer_id or staged.user_id

        # Determine participants and splits
        participant_ids: list[uuid.UUID] = []
        custom_splits: list[dict[str, Any]] = []

        if staged.split_data:
            for s in staged.split_data:
                u_str = s.get("user_id")
                if u_str:
                    try:
                        u_uuid = uuid.UUID(str(u_str))
                        participant_ids.append(u_uuid)
                        if "owed_amount" in s and s["owed_amount"] is not None:
                            custom_splits.append(
                                {"user_id": u_uuid, "owed_amount": Decimal(str(s["owed_amount"]))}
                            )
                    except ValueError:
                        pass

        # Default to caller and payer if participants empty
        if not participant_ids:
            participant_ids = [staged.user_id]
            if payer_id not in participant_ids:
                participant_ids.append(payer_id)

        # Create Expense
        expense = Expense(
            group_id=staged.group_id,
            description=staged.description,
            amount=staged.amount,
            currency=staged.currency,
            payer_id=payer_id,
            category=staged.category,
            expense_date=staged.expense_date,
            is_deleted=False,
        )
        session.add(expense)
        await session.flush()  # Populates expense.id

        # Calculate splits with exact paise distribution
        if len(custom_splits) == len(participant_ids) and sum(
            c["owed_amount"] for c in custom_splits
        ) == staged.amount:
            # Use pre-calculated exact splits
            for c in custom_splits:
                split = ExpenseSplit(
                    expense_id=expense.id,
                    user_id=c["user_id"],
                    owed_amount=c["owed_amount"],
                    split_type="EXACT",
                )
                session.add(split)
        else:
            # Equal split with deterministic paise allocation
            calculated = calculate_equal_splits(
                amount=staged.amount,
                participant_ids=participant_ids,
                payer_id=payer_id,
            )
            for u_id, owed in calculated:
                split = ExpenseSplit(
                    expense_id=expense.id,
                    user_id=u_id,
                    owed_amount=owed,
                    split_type="EQUAL",
                )
                session.add(split)

        staged.target_expense_id = expense.id
        staged.status = "APPROVED"
        staged.updated_at = datetime.now(UTC)
        approved_expense_id = expense.id

    elif staged.action_type == "UPDATE":
        if staged.target_expense_id:
            exp_result = await session.execute(
                select(Expense)
                .options(selectinload(Expense.splits))
                .where(Expense.id == staged.target_expense_id)
            )
            target_exp = exp_result.scalars().first()
            if target_exp:
                target_exp.description = staged.description
                target_exp.amount = staged.amount
                if staged.category:
                    target_exp.category = staged.category
                if staged.expense_date:
                    target_exp.expense_date = staged.expense_date
                target_exp.updated_at = datetime.now(UTC)

                # Rebalance splits if amount changed
                if target_exp.splits:
                    p_ids = [s.user_id for s in target_exp.splits]
                    recalculated = calculate_equal_splits(
                        amount=staged.amount,
                        participant_ids=p_ids,
                        payer_id=target_exp.payer_id,
                    )
                    for split in target_exp.splits:
                        for u_id, new_owed in recalculated:
                            if split.user_id == u_id:
                                split.owed_amount = new_owed
                                break

                approved_expense_id = target_exp.id

        staged.status = "APPROVED"
        staged.updated_at = datetime.now(UTC)

    elif staged.action_type == "DELETE":
        if staged.target_expense_id:
            exp_result = await session.execute(
                select(Expense).where(Expense.id == staged.target_expense_id)
            )
            target_exp = exp_result.scalars().first()
            if target_exp:
                target_exp.is_deleted = True
                target_exp.updated_at = datetime.now(UTC)
                approved_expense_id = target_exp.id

        staged.status = "APPROVED"
        staged.updated_at = datetime.now(UTC)

    return approved_expense_id


async def batch_approve_staged(
    session: AsyncSession,
    user_id: uuid.UUID,
    staged_ids: list[uuid.UUID] | None = None,
    group_id: uuid.UUID | None = None,
) -> BatchApproveResponse:
    """Atomically approve multiple or all pending staged items inside a single transaction."""
    query = select(StagedExpense).where(
        StagedExpense.user_id == user_id,
        StagedExpense.status == "PENDING",
    )
    if staged_ids:
        query = query.where(StagedExpense.id.in_(staged_ids))
    if group_id:
        query = query.where(StagedExpense.group_id == group_id)

    result = await session.execute(query)
    items = list(result.scalars().all())

    approved_ids: list[uuid.UUID] = []
    for item in items:
        exp_id = await approve_staged_item(session, item)
        if exp_id:
            approved_ids.append(exp_id)

    await session.commit()
    return BatchApproveResponse(
        approved_count=len(items),
        approved_expense_ids=approved_ids,
        message=f"Successfully approved {len(items)} staged expense(s).",
    )
