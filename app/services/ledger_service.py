"""Ledger service: split calculation (exact paise rounding) and balance resolution."""

import uuid
from collections import defaultdict
from decimal import ROUND_DOWN, Decimal

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.expense import Expense
from app.models.settlement import Settlement
from app.models.user import User
from app.schemas.ledger import BalanceSummary, UserBalance


def calculate_equal_splits(
    amount: Decimal,
    participant_ids: list[uuid.UUID],
    payer_id: uuid.UUID | None = None,
) -> list[tuple[uuid.UUID, Decimal]]:
    """
    Distribute amount equally among N participants with exact paise precision.

    Remainder paise are deterministically allocated to the payer (or first participant)
    such that: sum(owed_amounts) == amount exactly.
    """
    if not participant_ids:
        raise ValueError("At least one participant is required to split an expense.")

    n = len(participant_ids)
    total_paise = int((amount * 100).to_integral_value(rounding=ROUND_DOWN))
    base_paise = total_paise // n
    remainder_paise = total_paise % n

    priority_user = payer_id if payer_id in participant_ids else participant_ids[0]

    splits: list[tuple[uuid.UUID, Decimal]] = []
    paise_allocated = 0

    for user_id in participant_ids:
        assigned = base_paise
        if remainder_paise > 0:
            if user_id == priority_user or remainder_paise > 1:
                assigned += 1
                remainder_paise -= 1

        owed = Decimal(assigned) / Decimal(100)
        splits.append((user_id, owed))
        paise_allocated += assigned

    leftover = total_paise - paise_allocated
    if leftover != 0:
        for i, (u_id, owed) in enumerate(splits):
            if u_id == priority_user:
                splits[i] = (u_id, owed + (Decimal(leftover) / Decimal(100)))
                break

    return splits


def calculate_percentage_splits(
    amount: Decimal,
    participant_percentages: list[tuple[uuid.UUID, Decimal]],
    payer_id: uuid.UUID | None = None,
) -> list[tuple[uuid.UUID, Decimal]]:
    """
    Distribute amount according to percentages with exact paise precision (Hare-Niemeyer method).

    Guarantees:
    1. sum(owed_amounts) == amount exactly (zero floating loss).
    2. No participant is off by more than 1 paisa from their theoretical share.
    """
    if not participant_percentages:
        raise ValueError("At least one participant percentage is required.")

    total_pct = sum(pct for _, pct in participant_percentages)
    if abs(total_pct - Decimal("100.00")) > Decimal("0.05"):
        raise ValueError(f"Percentages must sum to 100%, got {total_pct}%")

    total_paise = int((amount * 100).to_integral_value(rounding=ROUND_DOWN))

    entries = []
    base_allocated = 0
    for idx, (user_id, pct) in enumerate(participant_percentages):
        exact_share = (Decimal(total_paise) * pct) / Decimal(100)
        base = int(exact_share.to_integral_value(rounding=ROUND_DOWN))
        remainder = exact_share - Decimal(base)
        base_allocated += base
        entries.append(
            {
                "idx": idx,
                "user_id": user_id,
                "pct": pct,
                "base": base,
                "remainder": remainder,
                "is_payer": (user_id == payer_id),
            }
        )

    shortfall = total_paise - base_allocated

    # Rank participants by:
    # 1. Highest remainder fraction
    # 2. Payer priority flag
    # 3. Original index
    entries.sort(key=lambda x: (x["remainder"], 1 if x["is_payer"] else 0), reverse=True)

    # Distribute the remaining shortfall paise to top ranked
    for i in range(shortfall):
        entries[i]["base"] += 1

    # Restore original ordering
    entries.sort(key=lambda x: x["idx"])

    result = [(e["user_id"], Decimal(e["base"]) / Decimal(100)) for e in entries]

    assert sum(owed for _, owed in result) == amount, "Split invariant violation!"
    return result


async def get_user_balances(
    session: AsyncSession,
    user_id: uuid.UUID,
    group_id: uuid.UUID | None = None,
) -> BalanceSummary:
    """
    Compute net balances for a user across all active expenses and settlements.

    Positive net_balance means others owe the user.
    Negative net_balance means the user owes others.
    """
    # 1. Expenses calculation
    query = (
        select(Expense)
        .options(selectinload(Expense.splits))
        .where(Expense.is_deleted.is_(False))
    )
    if group_id:
        query = query.where(Expense.group_id == group_id)

    result = await session.execute(query)
    expenses = result.scalars().all()

    # net_by_user[other_user_id] = net amount they owe to user_id
    net_by_user: dict[uuid.UUID, Decimal] = defaultdict(lambda: Decimal("0.00"))

    for exp in expenses:
        if exp.payer_id == user_id:
            # Caller paid: others owe caller their split amount
            for split in exp.splits:
                if split.user_id != user_id:
                    net_by_user[split.user_id] += split.owed_amount
        else:
            # Someone else paid: caller owes payer their split portion
            for split in exp.splits:
                if split.user_id == user_id:
                    net_by_user[exp.payer_id] -= split.owed_amount

    # 2. Settlements calculation (Settle Up / Direct Payments)
    settlements_query = select(Settlement).where(
        Settlement.is_deleted.is_(False),
        or_(Settlement.payer_id == user_id, Settlement.payee_id == user_id),
    )
    if group_id:
        settlements_query = settlements_query.where(Settlement.group_id == group_id)

    settlements_res = await session.execute(settlements_query)
    settlements = settlements_res.scalars().all()

    for s in settlements:
        if s.payer_id == user_id:
            # Caller paid payee money: reduces what caller owes payee (or increases what payee owes caller)
            net_by_user[s.payee_id] += s.amount
        elif s.payee_id == user_id:
            # Payer paid caller money: reduces what payer owes caller
            net_by_user[s.payer_id] -= s.amount

    # Fetch user names for friends in balance map
    other_user_ids = list(net_by_user.keys())
    user_names: dict[uuid.UUID, str] = {}
    if other_user_ids:
        users_result = await session.execute(
            select(User).where(User.id.in_(other_user_ids))
        )
        for u in users_result.scalars().all():
            user_names[u.id] = u.name

    total_owed_to_you = Decimal("0.00")
    total_you_owe = Decimal("0.00")
    user_balances: list[UserBalance] = []

    for other_id, net in net_by_user.items():
        if net > Decimal("0.00"):
            total_owed_to_you += net
        elif net < Decimal("0.00"):
            total_you_owe += abs(net)

        user_balances.append(
            UserBalance(
                user_id=other_id,
                user_name=user_names.get(other_id, "Unknown"),
                net_balance=net,
            )
        )

    user_balances.sort(key=lambda b: abs(b.net_balance), reverse=True)

    return BalanceSummary(
        total_owed_to_you=total_owed_to_you,
        total_you_owe=total_you_owe,
        net_total=total_owed_to_you - total_you_owe,
        balances=user_balances,
    )
