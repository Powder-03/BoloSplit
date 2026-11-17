"""Verified ledger expense endpoints (manual fallback, editing, deleting & querying)."""

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.database import get_session
from app.models.expense import Expense, ExpenseSplit
from app.models.user import User
from app.schemas.ledger import (
    ExpenseCreate,
    ExpenseRead,
    ExpenseUpdate,
    SelfExpenseCreate,
    SelfExpensesSummary,
)
from app.services.ledger_service import (
    calculate_equal_splits,
    get_self_expenses_summary,
)

router = APIRouter(prefix="/expenses", tags=["Verified Expenses Ledger"])


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


@router.get(
    "",
    response_model=list[ExpenseRead],
    summary="List verified expenses from main ledger",
)
async def list_expenses(
    group_id: uuid.UUID | None = Query(None, description="Filter by group ID"),
    scope: str | None = Query(
        None,
        description="Filter scope: 'self' (solo only), 'shared' (friends & groups), or 'all' (default)",
    ),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[ExpenseRead]:
    """Retrieve verified active expenses with splits, newest first."""
    other_split_subquery = select(ExpenseSplit.id).where(
        ExpenseSplit.expense_id == Expense.id,
        ExpenseSplit.user_id != current_user.id,
    )
    query = (
        select(Expense)
        .options(selectinload(Expense.splits))
        .where(Expense.is_deleted.is_(False))
        .order_by(Expense.expense_date.desc())
    )
    if group_id:
        query = query.where(Expense.group_id == group_id)

    if scope == "self":
        query = query.where(
            Expense.group_id.is_(None),
            Expense.payer_id == current_user.id,
            ~other_split_subquery.exists(),
        )
    elif scope == "shared":
        query = query.where(
            or_(
                Expense.group_id.is_not(None),
                other_split_subquery.exists(),
            )
        )

    query = query.offset(offset).limit(limit)
    result = await session.execute(query)
    expenses = result.scalars().all()
    return await _attach_payer_names(expenses, session)


@router.post(
    "/self",
    response_model=ExpenseRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a strictly solo / personal expense for self",
)
async def create_self_expense(
    payload: SelfExpenseCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ExpenseRead:
    """Create a 100% solo personal expense allocated entirely to the active user."""
    expense = Expense(
        group_id=None,
        description=payload.description,
        amount=payload.amount,
        currency=payload.currency,
        payer_id=current_user.id,
        category=payload.category or "general",
        expense_date=payload.expense_date or datetime.now(UTC),
        is_deleted=False,
    )
    session.add(expense)
    await session.flush()

    split = ExpenseSplit(
        expense_id=expense.id,
        user_id=current_user.id,
        owed_amount=payload.amount,
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
    res = ExpenseRead.model_validate(saved)
    res.payer_name = current_user.name
    return res


@router.get(
    "/self",
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
        .where(Expense.id == expense_id, Expense.is_deleted.is_(False))
    )
    result = await session.execute(query)
    expense = result.scalars().first()
    if not expense:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expense not found.",
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
    query = select(Expense).where(Expense.id == expense_id, Expense.is_deleted.is_(False))
    expense = (await session.execute(query)).scalars().first()
    if not expense:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Expense not found.")

    expense.is_deleted = True
    expense.updated_at = datetime.now(UTC)
    session.add(expense)
    await session.commit()
