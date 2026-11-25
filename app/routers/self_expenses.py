"""Strictly solo / personal expense CRUD endpoints (`/expenses/self`), backed by unified `Expense(is_personal=True)`."""

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_session
from app.models.expense import Expense
from app.models.user import User
from app.schemas.ledger import (
    SelfExpenseCreate,
    SelfExpenseRead,
    SelfExpensesSummary,
    SelfExpenseUpdate,
)
from app.services.ledger_service import get_self_expenses_summary

router = APIRouter(prefix="/expenses/self", tags=["Self / Personal Expenses"])


@router.get(
    "",
    response_model=SelfExpensesSummary,
    summary="Get strictly solo expenses, monthly spending & category breakdown for self",
)
async def get_self_expenses(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    category: str | None = Query(None, description="Filter by category (e.g. 'fuel', 'food')"),
    month: int | None = Query(None, ge=1, le=12, description="Filter by calendar month (1-12)"),
    year: int | None = Query(None, ge=2000, le=2100, description="Filter by calendar year"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> SelfExpensesSummary:
    """Retrieve personal ledger summary, monthly spend, category breakdown and solo transactions."""
    return await get_self_expenses_summary(
        session=session,
        user=current_user,
        limit=limit,
        offset=offset,
        category=category,
        month=month,
        year=year,
    )


@router.post(
    "",
    response_model=SelfExpenseRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a strictly solo / personal expense for self",
)
async def create_self_expense(
    payload: SelfExpenseCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> SelfExpenseRead:
    """Create a 100% solo personal expense (`is_personal=True`)."""
    expense = Expense(
        payer_id=current_user.id,
        description=payload.description.strip(),
        amount=payload.amount,
        currency=payload.currency,
        category=payload.category or "general",
        expense_date=payload.expense_date or datetime.now(UTC),
        is_personal=True,
        is_deleted=False,
    )
    session.add(expense)
    await session.commit()
    await session.refresh(expense)
    return SelfExpenseRead.model_validate(expense)


@router.get(
    "/{expense_id}",
    response_model=SelfExpenseRead,
    summary="Get single solo expense details",
)
async def get_self_expense_detail(
    expense_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> SelfExpenseRead:
    """Fetch a single solo expense owned by the active user."""
    expense = (
        await session.execute(
            select(Expense).where(
                Expense.id == expense_id,
                Expense.payer_id == current_user.id,
                Expense.is_personal.is_(True),
                Expense.is_deleted.is_(False),
            )
        )
    ).scalars().first()
    if not expense:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Self expense not found.")
    return SelfExpenseRead.model_validate(expense)


@router.patch(
    "/{expense_id}",
    response_model=SelfExpenseRead,
    summary="Update a solo expense",
)
async def update_self_expense(
    expense_id: uuid.UUID,
    payload: SelfExpenseUpdate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> SelfExpenseRead:
    """Update description, amount, category, or date on a solo expense."""
    expense = (
        await session.execute(
            select(Expense).where(
                Expense.id == expense_id,
                Expense.payer_id == current_user.id,
                Expense.is_personal.is_(True),
                Expense.is_deleted.is_(False),
            )
        )
    ).scalars().first()
    if not expense:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Self expense not found.")

    if payload.description is not None:
        cleaned = payload.description.strip()
        if not cleaned:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Description cannot be empty.",
            )
        expense.description = cleaned
    if payload.amount is not None:
        expense.amount = payload.amount
    if payload.category is not None:
        expense.category = payload.category
    if payload.expense_date is not None:
        expense.expense_date = payload.expense_date

    expense.updated_at = datetime.now(UTC)
    session.add(expense)
    await session.commit()
    await session.refresh(expense)
    return SelfExpenseRead.model_validate(expense)


@router.delete(
    "/{expense_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a solo expense",
)
async def delete_self_expense(
    expense_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    """Soft-delete a solo expense owned by the active user."""
    expense = (
        await session.execute(
            select(Expense).where(
                Expense.id == expense_id,
                Expense.payer_id == current_user.id,
                Expense.is_personal.is_(True),
                Expense.is_deleted.is_(False),
            )
        )
    ).scalars().first()
    if not expense:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Self expense not found.")

    expense.is_deleted = True
    expense.updated_at = datetime.now(UTC)
    session.add(expense)
    await session.commit()
