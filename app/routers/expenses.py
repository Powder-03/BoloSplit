"""Verified ledger expense endpoints (manual fallback, editing, deleting & querying)."""

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.database import get_session
from app.models.expense import Expense, ExpenseSplit
from app.models.user import User
from app.schemas.ledger import ExpenseCreate, ExpenseRead, ExpenseUpdate
from app.services.ledger_service import calculate_equal_splits

router = APIRouter(prefix="/expenses", tags=["Verified Expenses Ledger"])


@router.get(
    "",
    response_model=list[ExpenseRead],
    summary="List verified expenses from main ledger",
)
async def list_expenses(
    group_id: uuid.UUID | None = Query(None, description="Filter by group ID"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[ExpenseRead]:
    """Retrieve verified active expenses with splits, newest first."""
    query = (
        select(Expense)
        .options(selectinload(Expense.splits))
        .where(Expense.is_deleted.is_(False))
        .order_by(Expense.expense_date.desc())
        .offset(offset)
        .limit(limit)
    )
    if group_id:
        query = query.where(Expense.group_id == group_id)

    result = await session.execute(query)
    expenses = result.scalars().all()
    return [ExpenseRead.model_validate(e) for e in expenses]


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
    """Manual fallback form creation for an expense and its splits."""
    expense = Expense(
        group_id=payload.group_id,
        description=payload.description,
        amount=payload.amount,
        currency=payload.currency,
        payer_id=payload.payer_id,
        category=payload.category or "general",
        expense_date=payload.expense_date or datetime.now(UTC),
        is_deleted=False,
    )
    session.add(expense)
    await session.flush()

    if payload.splits:
        split_sum = sum(s.owed_amount for s in payload.splits)
        if split_sum != payload.amount:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Split amounts sum ({split_sum}) does not equal total amount ({payload.amount}).",
            )
        for s in payload.splits:
            split = ExpenseSplit(
                expense_id=expense.id,
                user_id=s.user_id,
                owed_amount=s.owed_amount,
                split_type=s.split_type,
                split_value=s.split_value,
            )
            session.add(split)
    else:
        participants = [current_user.id]
        if payload.payer_id not in participants:
            participants.append(payload.payer_id)

        calculated = calculate_equal_splits(
            amount=payload.amount,
            participant_ids=participants,
            payer_id=payload.payer_id,
        )
        for u_id, owed in calculated:
            split = ExpenseSplit(
                expense_id=expense.id,
                user_id=u_id,
                owed_amount=owed,
                split_type="EQUAL",
            )
            session.add(split)

    await session.commit()

    fetch_query = (
        select(Expense)
        .options(selectinload(Expense.splits))
        .where(Expense.id == expense.id)
    )
    saved = (await session.execute(fetch_query)).scalars().first()
    return ExpenseRead.model_validate(saved)


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
        .where(Expense.id == expense_id, Expense.is_deleted.is_(False))
    )
    result = await session.execute(query)
    expense = result.scalars().first()
    if not expense:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expense not found.",
        )
    return ExpenseRead.model_validate(expense)


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
    """Edit an existing expense amount or description directly from activity feed."""
    query = (
        select(Expense)
        .options(selectinload(Expense.splits))
        .where(Expense.id == expense_id, Expense.is_deleted.is_(False))
    )
    expense = (await session.execute(query)).scalars().first()
    if not expense:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Expense not found.")

    if payload.description is not None:
        expense.description = payload.description
    if payload.category is not None:
        expense.category = payload.category
    if payload.expense_date is not None:
        expense.expense_date = payload.expense_date

    if payload.amount is not None and payload.amount != expense.amount:
        expense.amount = payload.amount
        # Rebalance equal splits
        if expense.splits:
            p_ids = [s.user_id for s in expense.splits]
            recalculated = calculate_equal_splits(expense.amount, p_ids, payer_id=expense.payer_id)
            for split in expense.splits:
                for uid, owed in recalculated:
                    if split.user_id == uid:
                        split.owed_amount = owed
                        break

    expense.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(expense)
    return ExpenseRead.model_validate(expense)


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
    query = select(Expense).where(Expense.id == expense_id, Expense.is_deleted.is_(False))
    expense = (await session.execute(query)).scalars().first()
    if not expense:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Expense not found.")

    expense.is_deleted = True
    expense.updated_at = datetime.now(UTC)
    session.add(expense)
    await session.commit()
