"""Settlement endpoints for recording payments ('Settle Up')."""

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_session
from app.models.group import GroupMember
from app.models.settlement import Settlement
from app.models.user import User
from app.schemas.settlement import SettlementCreate, SettlementRead

router = APIRouter(prefix="/settlements", tags=["Settlements ('Settle Up')"])


@router.post(
    "",
    response_model=SettlementRead,
    status_code=status.HTTP_201_CREATED,
    summary="Record a settlement payment (Settle Up)",
)
async def create_settlement(
    payload: SettlementCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> SettlementRead:
    """Record a debt settlement payment (either paid by active user or received from a friend)."""
    actual_payer_id = payload.payer_id or current_user.id

    if actual_payer_id == payload.payee_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot settle up with yourself.",
        )

    if current_user.id not in (actual_payer_id, payload.payee_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You must be either the payer or the payee of the settlement.",
        )

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

    users_res = await session.execute(
        select(User).where(User.id.in_([actual_payer_id, payload.payee_id]))
    )
    users_by_id = {u.id: u for u in users_res.scalars().all()}

    if payload.payee_id not in users_by_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payee user not found.",
        )
    if actual_payer_id not in users_by_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payer user not found.",
        )

    settlement = Settlement(
        payer_id=actual_payer_id,
        payee_id=payload.payee_id,
        amount=payload.amount,
        currency=payload.currency,
        group_id=payload.group_id,
        payment_method=payload.payment_method,
        transaction_reference=payload.transaction_reference,
        notes=payload.notes,
        settlement_date=payload.settlement_date or datetime.now(UTC),
        is_deleted=False,
    )
    session.add(settlement)
    await session.commit()
    await session.refresh(settlement)

    return SettlementRead(
        id=settlement.id,
        payer_id=settlement.payer_id,
        payer_name=users_by_id[actual_payer_id].name,
        payee_id=settlement.payee_id,
        payee_name=users_by_id[payload.payee_id].name,
        amount=settlement.amount,
        currency=settlement.currency,
        group_id=settlement.group_id,
        payment_method=settlement.payment_method,
        transaction_reference=settlement.transaction_reference,
        notes=settlement.notes,
        settlement_date=settlement.settlement_date,
        is_deleted=settlement.is_deleted,
        created_at=settlement.created_at,
    )


@router.get(
    "",
    response_model=list[SettlementRead],
    summary="List settlements for active user",
)
async def list_settlements(
    friend_id: uuid.UUID | None = Query(None, description="Filter by friend ID"),
    group_id: uuid.UUID | None = Query(None, description="Filter by group ID"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[SettlementRead]:
    """Retrieve all non-deleted settlement transactions involving the active user."""
    query = (
        select(Settlement)
        .where(
            Settlement.is_deleted.is_(False),
            or_(Settlement.payer_id == current_user.id, Settlement.payee_id == current_user.id),
        )
        .order_by(Settlement.settlement_date.desc())
        .offset(offset)
        .limit(limit)
    )
    if friend_id:
        query = query.where(
            or_(
                (Settlement.payer_id == current_user.id) & (Settlement.payee_id == friend_id),
                (Settlement.payer_id == friend_id) & (Settlement.payee_id == current_user.id),
            )
        )
    if group_id:
        query = query.where(Settlement.group_id == group_id)

    result = await session.execute(query)
    settlements = result.scalars().all()

    user_ids = {s.payer_id for s in settlements} | {s.payee_id for s in settlements}
    names: dict[uuid.UUID, str] = {}
    if user_ids:
        u_res = await session.execute(select(User).where(User.id.in_(user_ids)))
        for u in u_res.scalars().all():
            names[u.id] = u.name

    return [
        SettlementRead(
            id=s.id,
            payer_id=s.payer_id,
            payer_name=names.get(s.payer_id),
            payee_id=s.payee_id,
            payee_name=names.get(s.payee_id),
            amount=s.amount,
            currency=s.currency,
            group_id=s.group_id,
            payment_method=s.payment_method,
            transaction_reference=s.transaction_reference,
            notes=s.notes,
            settlement_date=s.settlement_date,
            is_deleted=s.is_deleted,
            created_at=s.created_at,
        )
        for s in settlements
    ]


@router.delete(
    "/{settlement_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete / undo a settlement payment",
)
async def delete_settlement(
    settlement_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    """Soft delete a settlement to restore previous balances."""
    query = select(Settlement).where(
        Settlement.id == settlement_id,
        or_(Settlement.payer_id == current_user.id, Settlement.payee_id == current_user.id),
        Settlement.is_deleted.is_(False),
    )
    result = await session.execute(query)
    settlement = result.scalars().first()
    if not settlement:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Settlement not found.",
        )

    settlement.is_deleted = True
    settlement.updated_at = datetime.now(UTC)
    session.add(settlement)
    await session.commit()
