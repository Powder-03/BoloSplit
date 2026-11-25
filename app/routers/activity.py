"""Unified Activity Log endpoints (`GET /activity`)."""

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_session
from app.models.user import User
from app.schemas.ledger import ActivityLogRead
from app.services.activity_service import get_user_activity_feed

router = APIRouter(prefix="/activity", tags=["Activity Feed"])


@router.get(
    "",
    response_model=list[ActivityLogRead],
    summary="Get unified activity log feed (created, updated, deleted expenses & settlements)",
)
async def list_activity(
    entity_type: str | None = Query(
        None,
        description="Optional entity filter: EXPENSE or SETTLEMENT",
    ),
    event_type: str | None = Query(
        None,
        description="Optional event filter: EXPENSE_CREATED, EXPENSE_UPDATED, EXPENSE_DELETED, SETTLEMENT_CREATED, SETTLEMENT_DELETED",
    ),
    group_id: uuid.UUID | None = Query(None, description="Optional group ID filter"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[ActivityLogRead]:
    """Retrieve chronological activity events involving the active user."""
    return await get_user_activity_feed(
        session=session,
        current_user=current_user,
        entity_type=entity_type,
        event_type=event_type,
        group_id=group_id,
        limit=limit,
        offset=offset,
    )
