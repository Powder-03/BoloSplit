"""Verified ledger expense endpoints (manual fallback, editing, deleting & querying)."""

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.database import get_session
from app.models.expense import Expense, ExpenseSplit
from app.models.group import GroupMember
from app.models.user import User
from app.schemas.ledger import (
    ExpenseCreate,
    ExpenseRead,
    ExpenseUpdate,
)
from app.services.ledger_service import (
    calculate_equal_splits,
    recalculate_splits_on_amount_change,
)

router = APIRouter(prefix="/expenses", tags=["Shared Expenses Ledger"])


async def _attach_payer_names(expenses: list[Expense], session: AsyncSession) -> list[ExpenseRead]:
    """Helper to enrich ExpenseRead objects with payer names."""
    payer_ids = list({e.payer_id for e in expenses})
    user_names: dict[uuid.UUID, str] = {}
    if payer_ids:
        users_res = await session.execute(select(User).where(User.id.in_(payer_ids)))
        for u in users_res.scalars().all():
            user_names[u.id] = u.name

    reads: list[ExpenseRead] = []
    for e in expenses:
        model = ExpenseRead.model_validate(e)
        model.payer_name = user_names.get(e.payer_id, "Someone")
        reads.append(model)
    return reads


def _user_accessible_expense_clause(user_id: uuid.UUID):
    """SQLAlchemy filter restricting expenses to those the user paid, shares, or belongs to via group."""
    caller_split_subquery = select(ExpenseSplit.id).where(
        ExpenseSplit.expense_id == Expense.id,
        ExpenseSplit.user_id == user_id,
    )
    caller_group_subquery = select(GroupMember.id).where(
        GroupMember.group_id == Expense.group_id,
        GroupMember.user_id == user_id,
    )
    return or_(
        Expense.payer_id == user_id,
        caller_split_subquery.exists(),
        caller_group_subquery.exists(),
    )


async def _user_can_access_expense(
    session: AsyncSession, expense: Expense, user_id: uuid.UUID
) -> bool:
    """Verify whether user_id is payer, split participant, or group member of expense."""
    if expense.payer_id == user_id:
        return True
    split_match = (
        await session.execute(
            select(ExpenseSplit.id).where(
                ExpenseSplit.expense_id == expense.id,
                ExpenseSplit.user_id == user_id,
            )
        )
    ).scalars().first()
    if split_match:
        return True
    if expense.group_id is not None:
        mem_match = (
            await session.execute(
                select(GroupMember.id).where(
                    GroupMember.group_id == expense.group_id,
                    GroupMember.user_id == user_id,
                )
            )
        ).scalars().first()
        if mem_match:
            return True
    return False


@router.get(
    "",
    response_model=list[ExpenseRead],
    summary="List shared expenses from the verified ledger",
)
async def list_expenses(
    group_id: uuid.UUID | None = Query(None, description="Filter by group ID"),
    scope: str | None = Query(
        None,
        description="Deprecated filter ('shared' / 'all' return shared expenses; use /expenses/self for personal expenses).",
    ),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[ExpenseRead]:
    """Retrieve verified active shared expenses involving the active user, newest first."""
    if scope == "self":
        return []

    query = (
        select(Expense)
        .options(selectinload(Expense.splits))
        .where(
            Expense.is_deleted.is_(False),
            Expense.is_personal.is_(False),
            _user_accessible_expense_clause(current_user.id),
        )
        .order_by(Expense.expense_date.desc(), Expense.created_at.desc())
    )
    if group_id:
        query = query.where(Expense.group_id == group_id)

    query = query.offset(offset).limit(limit)
    result = await session.execute(query)
    expenses = result.scalars().all()
    return await _attach_payer_names(expenses, session)


@router.post(
    "",
    response_model=ExpenseRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create verified expense directly (manual mobile fallback)",
)
async def create_expense_manual(
    payload: ExpenseCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ExpenseRead:
    """Manual fallback form creation for a shared expense and its splits."""
    if payload.group_id is not None:
        mem_check = (
            await session.execute(
                select(GroupMember.id).where(
                    GroupMember.group_id == payload.group_id,
                    GroupMember.user_id == current_user.id,
                )
            )
        ).scalars().first()
        if not mem_check:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Group not found or you are not a member of this group.",
            )

    expense = Expense(
        group_id=payload.group_id,
        description=payload.description,
        amount=payload.amount,
        currency=payload.currency,
        payer_id=payload.payer_id,
        category=payload.category or "general",
        expense_date=payload.expense_date or datetime.now(UTC),
        is_personal=False,
        is_deleted=False,
    )
    session.add(expense)
    await session.flush()

    if payload.splits:
        # Merge duplicate user_id entries if caller supplied any
        merged_splits: dict[uuid.UUID, tuple[Decimal, str, Decimal | None]] = {}
        for s in payload.splits:
            if s.user_id in merged_splits:
                prev_owed, prev_type, prev_val = merged_splits[s.user_id]
                next_val = (prev_val + s.split_value) if (prev_val is not None and s.split_value is not None) else prev_val
                merged_splits[s.user_id] = (prev_owed + s.owed_amount, prev_type, next_val)
            else:
                merged_splits[s.user_id] = (s.owed_amount, s.split_type, s.split_value)

        split_sum = sum(owed for owed, _, _ in merged_splits.values())
        if split_sum != payload.amount:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Split amounts sum ({split_sum}) does not equal total amount ({payload.amount}).",
            )
        for uid, (owed, stype, sval) in merged_splits.items():
            session.add(
                ExpenseSplit(
                    expense_id=expense.id,
                    user_id=uid,
                    owed_amount=owed,
                    split_type=stype,
                    split_value=sval,
                )
            )
    else:
        participants: list[uuid.UUID] = []
        if payload.group_id is not None:
            grp_mem_res = await session.execute(
                select(GroupMember.user_id).where(GroupMember.group_id == payload.group_id)
            )
            participants = list(grp_mem_res.scalars().all())

        if not participants:
            participants = [current_user.id]
            if payload.payer_id not in participants:
                participants.append(payload.payer_id)

        calculated = calculate_equal_splits(
            amount=payload.amount,
            participant_ids=participants,
            payer_id=payload.payer_id,
        )
        for u_id, owed in calculated:
            session.add(
                ExpenseSplit(
                    expense_id=expense.id,
                    user_id=u_id,
                    owed_amount=owed,
                    split_type="EQUAL",
                )
            )

    await session.commit()

    fetch_query = (
        select(Expense)
        .options(selectinload(Expense.splits))
        .where(Expense.id == expense.id)
    )
    saved = (await session.execute(fetch_query)).scalars().first()
    res = ExpenseRead.model_validate(saved)
    payer_user = (await session.execute(select(User).where(User.id == saved.payer_id))).scalars().first()
    if payer_user:
        res.payer_name = payer_user.name
    return res


@router.get(
    "/{expense_id}",
    response_model=ExpenseRead,
    summary="Get single expense details with splits",
)
async def get_expense_detail(
    expense_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ExpenseRead:
    """Fetch expense and participant splits by ID."""
    query = (
        select(Expense)
        .options(selectinload(Expense.splits))
        .where(
            Expense.id == expense_id,
            Expense.is_deleted.is_(False),
            Expense.is_personal.is_(False),
        )
    )
    result = await session.execute(query)
    expense = result.scalars().first()
    if not expense:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expense not found.",
        )
    if not await _user_can_access_expense(session, expense, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this expense.",
        )
    payer_user = (await session.execute(select(User).where(User.id == expense.payer_id))).scalars().first()
    res = ExpenseRead.model_validate(expense)
    if payer_user:
        res.payer_name = payer_user.name
    return res


@router.patch(
    "/{expense_id}",
    response_model=ExpenseRead,
    summary="Edit an expense directly from Activity Feed",
)
async def update_expense(
    expense_id: uuid.UUID,
    payload: ExpenseUpdate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ExpenseRead:
    """Edit an existing expense amount or description directly from activity feed, preserving split proportions."""
    query = (
        select(Expense)
        .options(selectinload(Expense.splits))
        .where(
            Expense.id == expense_id,
            Expense.is_deleted.is_(False),
            Expense.is_personal.is_(False),
        )
    )
    expense = (await session.execute(query)).scalars().first()
    if not expense:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Expense not found.")
    if not await _user_can_access_expense(session, expense, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this expense.",
        )

    if payload.description is not None:
        expense.description = payload.description
    if payload.category is not None:
        expense.category = payload.category
    if payload.expense_date is not None:
        expense.expense_date = payload.expense_date

    if payload.amount is not None and payload.amount != expense.amount:
        expense.amount = payload.amount
        if expense.splits:
            recalculate_splits_on_amount_change(
                new_amount=expense.amount,
                existing_splits=expense.splits,
                payer_id=expense.payer_id,
            )

    expense.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(expense)
    payer_user = (await session.execute(select(User).where(User.id == expense.payer_id))).scalars().first()
    res = ExpenseRead.model_validate(expense)
    if payer_user:
        res.payer_name = payer_user.name
    return res


@router.delete(
    "/{expense_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an expense directly from Activity Feed",
)
async def delete_expense(
    expense_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    """Soft delete an expense from the verified ledger."""
    query = select(Expense).where(
        Expense.id == expense_id,
        Expense.is_deleted.is_(False),
        Expense.is_personal.is_(False),
    )
    expense = (await session.execute(query)).scalars().first()
    if not expense:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Expense not found.")
    if not await _user_can_access_expense(session, expense, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this expense.",
        )

    expense.is_deleted = True
    expense.updated_at = datetime.now(UTC)
    session.add(expense)
    await session.commit()
