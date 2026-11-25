"""Service for recording and querying unified audit activity logs."""

import uuid
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.activity import ActivityLog, ActivityLogAudience
from app.models.expense import Expense
from app.models.group import Group, GroupMember
from app.models.settlement import Settlement
from app.models.user import Friendship, User
from app.schemas.ledger import ActivityLogRead

EVENT_EXPENSE_CREATED = "EXPENSE_CREATED"
EVENT_EXPENSE_UPDATED = "EXPENSE_UPDATED"
EVENT_EXPENSE_DELETED = "EXPENSE_DELETED"
EVENT_SETTLEMENT_CREATED = "SETTLEMENT_CREATED"
EVENT_SETTLEMENT_DELETED = "SETTLEMENT_DELETED"


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize naive datetimes (e.g. from SQLite) to timezone-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


async def _next_activity_timestamp(session: AsyncSession) -> datetime:
    """Return a strictly increasing UTC timestamp for live activity events."""
    now = datetime.now(UTC)
    latest_raw = (await session.execute(select(func.max(ActivityLog.created_at)))).scalar_one_or_none()
    if latest_raw is None:
        return now
    latest = _ensure_utc(latest_raw)
    if latest >= now:
        return latest + timedelta(microseconds=1)
    return now


async def _add_audiences(
    session: AsyncSession,
    activity_id: uuid.UUID,
    user_ids: Iterable[uuid.UUID],
) -> None:
    """Insert deduplicated audience rows for an activity entry."""
    for uid in sorted(set(user_ids), key=str):
        session.add(ActivityLogAudience(activity_id=activity_id, user_id=uid))


async def record_expense_activity(
    session: AsyncSession,
    expense: Expense,
    actor_id: uuid.UUID,
    event_type: str,
    split_user_ids: Iterable[uuid.UUID] | None = None,
    metadata: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> ActivityLog | None:
    """
    Record a shared expense lifecycle event (`EXPENSE_CREATED`, `EXPENSE_UPDATED`, `EXPENSE_DELETED`).

    Personal/solo expenses (`is_personal=True`) are intentionally excluded from the shared activity feed.
    """
    if expense.is_personal:
        return None

    audience_ids: set[uuid.UUID] = {actor_id, expense.payer_id}
    if split_user_ids:
        audience_ids.update(split_user_ids)
    elif hasattr(expense, "splits") and expense.splits:
        audience_ids.update(sp.user_id for sp in expense.splits)

    if expense.group_id is not None:
        mem_rows = (
            await session.execute(
                select(GroupMember.user_id).where(GroupMember.group_id == expense.group_id)
            )
        ).scalars().all()
        audience_ids.update(mem_rows)

    event_ts = _ensure_utc(created_at) if created_at is not None else await _next_activity_timestamp(session)
    log = ActivityLog(
        event_type=event_type,
        entity_type="EXPENSE",
        entity_id=expense.id,
        actor_id=actor_id,
        group_id=expense.group_id,
        description=expense.description,
        amount=Decimal(str(expense.amount)).quantize(Decimal("0.01")),
        currency=expense.currency or "INR",
        category=expense.category or "general",
        payer_id=expense.payer_id,
        payee_id=None,
        metadata_json=dict(metadata or {}),
        created_at=event_ts,
    )
    session.add(log)
    await session.flush()
    await _add_audiences(session, log.id, audience_ids)
    return log


async def record_settlement_activity(
    session: AsyncSession,
    settlement: Settlement,
    actor_id: uuid.UUID,
    event_type: str,
    metadata: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> ActivityLog:
    """Record a settlement lifecycle event (`SETTLEMENT_CREATED` or `SETTLEMENT_DELETED`)."""
    audience_ids: set[uuid.UUID] = {actor_id, settlement.payer_id, settlement.payee_id}

    merged_meta: dict[str, Any] = {
        "payment_method": settlement.payment_method,
    }
    if settlement.notes:
        merged_meta["notes"] = settlement.notes
    if metadata:
        merged_meta.update(metadata)

    event_ts = _ensure_utc(created_at) if created_at is not None else await _next_activity_timestamp(session)
    description = settlement.notes.strip() if settlement.notes and settlement.notes.strip() else f"Settled via {settlement.payment_method}"

    log = ActivityLog(
        event_type=event_type,
        entity_type="SETTLEMENT",
        entity_id=settlement.id,
        actor_id=actor_id,
        group_id=settlement.group_id,
        description=description[:255],
        amount=Decimal(str(settlement.amount)).quantize(Decimal("0.01")),
        currency=settlement.currency or "INR",
        category=None,
        payer_id=settlement.payer_id,
        payee_id=settlement.payee_id,
        metadata_json=merged_meta,
        created_at=event_ts,
    )
    session.add(log)
    await session.flush()
    await _add_audiences(session, log.id, audience_ids)
    return log


async def backfill_missing_activity_logs(session: AsyncSession) -> int:
    """
    Idempotently backfill `EXPENSE_CREATED` and `SETTLEMENT_CREATED` logs
    for pre-existing active records that predated the `activity_logs` table.
    """
    created_count = 0

    existing_exp_ids = set(
        (
            await session.execute(
                select(ActivityLog.entity_id).where(ActivityLog.entity_type == "EXPENSE")
            )
        )
        .scalars()
        .all()
    )
    expenses = (
        await session.execute(
            select(Expense)
            .options(selectinload(Expense.splits))
            .where(Expense.is_personal.is_(False), Expense.is_deleted.is_(False))
        )
    ).scalars().all()
    for exp in expenses:
        if exp.id not in existing_exp_ids:
            await record_expense_activity(
                session=session,
                expense=exp,
                actor_id=exp.payer_id,
                event_type=EVENT_EXPENSE_CREATED,
                split_user_ids=[sp.user_id for sp in exp.splits],
                metadata={"backfilled": True},
                created_at=exp.created_at,
            )
            created_count += 1

    existing_set_ids = set(
        (
            await session.execute(
                select(ActivityLog.entity_id).where(ActivityLog.entity_type == "SETTLEMENT")
            )
        )
        .scalars()
        .all()
    )
    settlements = (
        await session.execute(select(Settlement).where(Settlement.is_deleted.is_(False)))
    ).scalars().all()
    for stl in settlements:
        if stl.id not in existing_set_ids:
            await record_settlement_activity(
                session=session,
                settlement=stl,
                actor_id=stl.payer_id,
                event_type=EVENT_SETTLEMENT_CREATED,
                metadata={"backfilled": True},
                created_at=stl.created_at,
            )
            created_count += 1

    return created_count


def _format_amount(amount: Decimal, currency: str) -> str:
    symbol = "₹" if (currency or "INR").upper() == "INR" else f"{currency} "
    return f"{symbol}{Decimal(str(amount)).quantize(Decimal('0.01')):.2f}"


def _build_summary_text(
    log: ActivityLog,
    current_user_id: uuid.UUID,
    display_names: dict[uuid.UUID, str],
) -> str:
    actor_label = "You" if log.actor_id == current_user_id else display_names.get(log.actor_id, "Someone")
    payer_label = (
        "You"
        if log.payer_id == current_user_id
        else display_names.get(log.payer_id, "Someone") if log.payer_id else "Someone"
    )
    payee_label = (
        "you"
        if log.payee_id == current_user_id
        else display_names.get(log.payee_id, "friend") if log.payee_id else "friend"
    )
    amount_str = _format_amount(log.amount, log.currency)
    event = log.event_type.upper()

    if event == EVENT_EXPENSE_CREATED:
        return f"{actor_label} added '{log.description}' ({amount_str})"
    if event == EVENT_EXPENSE_UPDATED:
        prev_amt = (log.metadata_json or {}).get("previous_amount")
        if prev_amt is not None and str(prev_amt) != str(log.amount):
            prev_str = _format_amount(Decimal(str(prev_amt)), log.currency)
            return f"{actor_label} updated '{log.description}' ({prev_str} → {amount_str})"
        return f"{actor_label} updated '{log.description}' ({amount_str})"
    if event == EVENT_EXPENSE_DELETED:
        return f"{actor_label} deleted '{log.description}' ({amount_str})"
    if event == EVENT_SETTLEMENT_CREATED:
        return f"{payer_label} paid {payee_label} {amount_str}"
    if event == EVENT_SETTLEMENT_DELETED:
        payee_display = "You" if log.payee_id == current_user_id else display_names.get(log.payee_id, "friend")
        return f"{actor_label} undid settlement ({payer_label} → {payee_display}, {amount_str})"
    return f"{actor_label}: {log.description}"


async def get_user_activity_feed(
    session: AsyncSession,
    current_user: User,
    entity_type: str | None = None,
    event_type: str | None = None,
    group_id: uuid.UUID | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[ActivityLogRead]:
    """Retrieve personalized chronological activity log items visible to `current_user`."""
    stmt = (
        select(ActivityLog)
        .join(ActivityLogAudience, ActivityLogAudience.activity_id == ActivityLog.id)
        .where(ActivityLogAudience.user_id == current_user.id)
    )

    if entity_type:
        stmt = stmt.where(ActivityLog.entity_type == entity_type.strip().upper())
    if event_type:
        stmt = stmt.where(ActivityLog.event_type == event_type.strip().upper())
    if group_id:
        stmt = stmt.where(ActivityLog.group_id == group_id)

    stmt = (
        stmt.order_by(ActivityLog.created_at.desc(), ActivityLog.id.desc())
        .offset(offset)
        .limit(limit)
    )
    logs = (await session.execute(stmt)).scalars().all()
    if not logs:
        return []

    user_ids: set[uuid.UUID] = set()
    group_ids: set[uuid.UUID] = set()
    expense_ids: set[uuid.UUID] = set()
    settlement_ids: set[uuid.UUID] = set()

    for log in logs:
        user_ids.add(log.actor_id)
        if log.payer_id:
            user_ids.add(log.payer_id)
        if log.payee_id:
            user_ids.add(log.payee_id)
        if log.group_id:
            group_ids.add(log.group_id)
        if log.entity_type == "EXPENSE":
            expense_ids.add(log.entity_id)
        elif log.entity_type == "SETTLEMENT":
            settlement_ids.add(log.entity_id)

    display_names: dict[uuid.UUID, str] = {}
    if user_ids:
        users = (await session.execute(select(User).where(User.id.in_(user_ids)))).scalars().all()
        for u in users:
            display_names[u.id] = u.name

        ships = (
            await session.execute(
                select(Friendship).where(
                    Friendship.user_id == current_user.id,
                    Friendship.friend_id.in_(user_ids),
                )
            )
        ).scalars().all()
        for ship in ships:
            if ship.nickname and ship.nickname.strip():
                display_names[ship.friend_id] = ship.nickname.strip()

    group_names: dict[uuid.UUID, str] = {}
    if group_ids:
        groups = (await session.execute(select(Group).where(Group.id.in_(group_ids)))).scalars().all()
        for g in groups:
            group_names[g.id] = g.name

    deleted_expenses: set[uuid.UUID] = set()
    if expense_ids:
        exp_rows = (
            await session.execute(
                select(Expense.id, Expense.is_deleted).where(Expense.id.in_(expense_ids))
            )
        ).all()
        for eid, is_del in exp_rows:
            if is_del:
                deleted_expenses.add(eid)

    deleted_settlements: set[uuid.UUID] = set()
    if settlement_ids:
        stl_rows = (
            await session.execute(
                select(Settlement.id, Settlement.is_deleted).where(Settlement.id.in_(settlement_ids))
            )
        ).all()
        for sid, is_del in stl_rows:
            if is_del:
                deleted_settlements.add(sid)

    items: list[ActivityLogRead] = []
    for log in logs:
        is_deleted = (
            (log.entity_type == "EXPENSE" and log.entity_id in deleted_expenses)
            or (log.entity_type == "SETTLEMENT" and log.entity_id in deleted_settlements)
            or log.event_type in (EVENT_EXPENSE_DELETED, EVENT_SETTLEMENT_DELETED)
        )
        items.append(
            ActivityLogRead(
                id=log.id,
                event_type=log.event_type,
                entity_type=log.entity_type,
                entity_id=log.entity_id,
                actor_id=log.actor_id,
                actor_name=display_names.get(log.actor_id, "Someone"),
                is_actor_you=(log.actor_id == current_user.id),
                group_id=log.group_id,
                group_name=group_names.get(log.group_id) if log.group_id else None,
                description=log.description,
                amount=log.amount,
                currency=log.currency,
                category=log.category,
                payer_id=log.payer_id,
                payer_name=display_names.get(log.payer_id) if log.payer_id else None,
                payee_id=log.payee_id,
                payee_name=display_names.get(log.payee_id) if log.payee_id else None,
                summary_text=_build_summary_text(log, current_user.id, display_names),
                is_entity_deleted=is_deleted,
                metadata=log.metadata_json or {},
                created_at=_ensure_utc(log.created_at),
            )
        )

    return items
