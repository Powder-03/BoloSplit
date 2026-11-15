"""Staging queue management and atomic approval endpoints."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_session
from app.models.staging import StagedExpense
from app.models.user import User
from app.schemas.staging import (
    BatchApproveRequest,
    BatchApproveResponse,
    StagedExpenseRead,
    StagedExpenseUpdate,
)
from app.services.staging_service import (
    approve_staged_item,
    batch_approve_staged,
    get_staged_expenses,
    update_staged_expense,
)

router = APIRouter(prefix="/staging", tags=["Staging Review Queue"])


@router.get(
    "",
    response_model=list[StagedExpenseRead],
    summary="Fetch pending staged expense review stack",
)
async def list_pending_staged_expenses(
    group_id: uuid.UUID | None = Query(None, description="Filter by group ID"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[StagedExpenseRead]:
    """
    Fetch all voice-staged items waiting for user verification and approval.

    Items are sorted newest first.
    """
    items = await get_staged_expenses(
        session=session,
        user_id=current_user.id,
        status="PENDING",
        group_id=group_id,
    )
    return [StagedExpenseRead.model_validate(item) for item in items]


@router.patch(
    "/{staged_id}",
    response_model=StagedExpenseRead,
    summary="Edit fields on an unapproved staged review card",
)
async def patch_staged_expense(
    staged_id: uuid.UUID,
    payload: StagedExpenseUpdate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> StagedExpenseRead:
    """Allow user to manually adjust amount, category, or splits before approving."""
    update_dict = payload.model_dump(exclude_unset=True)
    updated = await update_staged_expense(
        session=session,
        staged_id=staged_id,
        user_id=current_user.id,
        update_data=update_dict,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Staged expense item not found or already processed.",
        )
    return StagedExpenseRead.model_validate(updated)


@router.post(
    "/{staged_id}/approve",
    response_model=StagedExpenseRead,
    summary="Atomically approve a single staged item to the verified ledger",
)
async def approve_single_staged_expense(
    staged_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> StagedExpenseRead:
    """
    Atomically promote a staged proposal to the main verified ledger.

    - CREATE: Inserts Expense + ExpenseSplit rows and marks StagedExpense APPROVED.
    - UPDATE: Modifies verified expense and marks StagedExpense APPROVED.
    - DELETE: Soft-deletes target expense and marks StagedExpense APPROVED.
    """
    query = select(StagedExpense).where(
        StagedExpense.id == staged_id,
        StagedExpense.user_id == current_user.id,
        StagedExpense.status == "PENDING",
    )
    result = await session.execute(query)
    staged = result.scalars().first()
    if not staged:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Staged item not found or already approved.",
        )

    await approve_staged_item(session, staged)
    await session.commit()
    await session.refresh(staged)

    return StagedExpenseRead.model_validate(staged)


@router.post(
    "/approve-all",
    response_model=BatchApproveResponse,
    summary="Atomically approve all pending items in the review queue",
)
async def approve_all_staged_expenses(
    payload: BatchApproveRequest | None = None,
    group_id: uuid.UUID | None = Query(None, description="Optional group filter"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> BatchApproveResponse:
    """
    Batch promote all (or specified) staged items to the main ledger in a single transaction.
    """
    ids_to_approve = payload.staged_ids if payload else None
    return await batch_approve_staged(
        session=session,
        user_id=current_user.id,
        staged_ids=ids_to_approve,
        group_id=group_id,
    )


@router.delete(
    "/{staged_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Reject/discard an unapproved staged review item",
)
async def reject_staged_expense(
    staged_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    """Reject and discard an incorrect voice proposal without modifying the ledger."""
    query = select(StagedExpense).where(
        StagedExpense.id == staged_id,
        StagedExpense.user_id == current_user.id,
        StagedExpense.status == "PENDING",
    )
    result = await session.execute(query)
    staged = result.scalars().first()
    if not staged:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Staged item not found or already processed.",
        )

    staged.status = "REJECTED"
    session.add(staged)
    await session.commit()
