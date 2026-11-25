"""Ledger service: unified split engine (EQUAL, EXACT, PERCENTAGE, SHARES) and SQL-first balance resolution."""

import uuid
import zoneinfo
from collections import defaultdict
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.expense import Expense, ExpenseSplit
from app.models.settlement import Settlement
from app.models.user import Friendship, User
from app.schemas.ledger import (
    BalanceSummary,
    SelfCategoryStat,
    SelfExpenseRead,
    SelfExpensesSummary,
    UserBalance,
)

IST = zoneinfo.ZoneInfo("Asia/Kolkata")


def calculate_equal_splits(
    amount: Decimal,
    participant_ids: list[uuid.UUID],
    payer_id: uuid.UUID | None = None,
) -> list[tuple[uuid.UUID, Decimal]]:
    """
    Distribute amount equally among unique participants with exact paise precision.

    Remainder paise are deterministically allocated to the payer (or first participant)
    such that: sum(owed_amounts) == amount exactly.
    """
    # Deduplicate participant_ids while preserving order
    unique_ids = list(dict.fromkeys(participant_ids))
    if not unique_ids:
        raise ValueError("At least one participant is required to split an expense.")

    n = len(unique_ids)
    total_paise = int((amount * 100).to_integral_value(rounding=ROUND_DOWN))
    base_paise = total_paise // n
    remainder_paise = total_paise % n

    priority_user = payer_id if payer_id in unique_ids else unique_ids[0]

    # Sort indices so priority_user gets the first remainder paisa, followed by others in order
    ordered_indices = sorted(
        range(n),
        key=lambda i: (0 if unique_ids[i] == priority_user else 1, i),
    )
    extra_Recipients = set(ordered_indices[:remainder_paise])

    splits: list[tuple[uuid.UUID, Decimal]] = []
    for i, user_id in enumerate(unique_ids):
        assigned = base_paise + (1 if i in extra_Recipients else 0)
        splits.append((user_id, Decimal(assigned) / Decimal(100)))

    return splits


def calculate_share_splits(
    amount: Decimal,
    participant_shares: list[tuple[uuid.UUID, Decimal]],
    payer_id: uuid.UUID | None = None,
) -> list[tuple[uuid.UUID, Decimal]]:
    """
    Distribute amount proportionally according to relative weights/shares (Hare-Niemeyer method).
    """
    if not participant_shares:
        raise ValueError("At least one participant share is required.")

    # Combine duplicate user_ids by summing their shares
    merged: dict[uuid.UUID, Decimal] = {}
    for uid, weight in participant_shares:
        if weight < Decimal("0"):
            raise ValueError("Share weights cannot be negative.")
        merged[uid] = merged.get(uid, Decimal("0")) + weight

    total_weight = sum(merged.values())
    if total_weight <= Decimal("0"):
        raise ValueError("Total share weight must be greater than zero.")

    total_paise = int((amount * 100).to_integral_value(rounding=ROUND_DOWN))
    entries = []
    base_allocated = 0

    for idx, (user_id, weight) in enumerate(merged.items()):
        exact_share = (Decimal(total_paise) * weight) / total_weight
        base = int(exact_share.to_integral_value(rounding=ROUND_DOWN))
        remainder = exact_share - Decimal(base)
        base_allocated += base
        entries.append(
            {
                "idx": idx,
                "user_id": user_id,
                "base": base,
                "remainder": remainder,
                "is_payer": (user_id == payer_id),
            }
        )

    shortfall = total_paise - base_allocated
    entries.sort(key=lambda x: (x["remainder"], 1 if x["is_payer"] else 0, -x["idx"]), reverse=True)
    for i in range(shortfall):
        entries[i]["base"] += 1

    entries.sort(key=lambda x: x["idx"])
    result = [(e["user_id"], Decimal(e["base"]) / Decimal(100)) for e in entries]
    assert sum(owed for _, owed in result) == amount, "Split invariant violation!"
    return result


def calculate_percentage_splits(
    amount: Decimal,
    participant_percentages: list[tuple[uuid.UUID, Decimal]],
    payer_id: uuid.UUID | None = None,
) -> list[tuple[uuid.UUID, Decimal]]:
    """
    Distribute amount according to percentages with exact paise precision (Hare-Niemeyer method).
    """
    if not participant_percentages:
        raise ValueError("At least one participant percentage is required.")

    merged: dict[uuid.UUID, Decimal] = {}
    for uid, pct in participant_percentages:
        merged[uid] = merged.get(uid, Decimal("0")) + pct

    total_pct = sum(merged.values())
    if abs(total_pct - Decimal("100.00")) > Decimal("0.05"):
        raise ValueError(f"Percentages must sum to 100%, got {total_pct}%")

    return calculate_share_splits(amount, list(merged.items()), payer_id=payer_id)


def calculate_exact_splits(
    amount: Decimal,
    participant_amounts: list[tuple[uuid.UUID, Decimal]],
) -> list[tuple[uuid.UUID, Decimal]]:
    """
    Validate and merge exact currency splits such that sum(owed_amounts) == amount.
    """
    if not participant_amounts:
        raise ValueError("At least one participant exact amount is required.")

    merged: dict[uuid.UUID, Decimal] = {}
    for uid, exact_amt in participant_amounts:
        q_amt = exact_amt.quantize(Decimal("0.01"))
        if q_amt < Decimal("0.00"):
            raise ValueError("Exact split amount cannot be negative.")
        merged[uid] = merged.get(uid, Decimal("0.00")) + q_amt

    total_exact = sum(merged.values())
    if total_exact != amount.quantize(Decimal("0.01")):
        raise ValueError(f"Exact split sum ({total_exact}) does not equal total amount ({amount}).")

    return list(merged.items())


def compute_splits(
    amount: Decimal,
    split_type: str,
    participants: list[tuple[uuid.UUID, Decimal | None]],
    payer_id: uuid.UUID | None = None,
) -> list[tuple[uuid.UUID, Decimal, str, Decimal | None]]:
    """
    Unified split engine supporting EQUAL, PERCENTAGE, EXACT, and SHARES.

    Returns list of (user_id, owed_amount, split_type, split_value).
    """
    stype = (split_type or "EQUAL").strip().upper()
    if stype == "PERCENTAGE":
        pct_inputs: list[tuple[uuid.UUID, Decimal]] = []
        merged_pcts: dict[uuid.UUID, Decimal] = {}
        for uid, val in participants:
            if val is None:
                raise ValueError("Percentage value missing for participant.")
            d_val = Decimal(str(val)).quantize(Decimal("0.01"))
            merged_pcts[uid] = merged_pcts.get(uid, Decimal("0.00")) + d_val
        pct_inputs = list(merged_pcts.items())
        calculated = calculate_percentage_splits(amount, pct_inputs, payer_id=payer_id)
        return [(uid, owed, "PERCENTAGE", merged_pcts[uid]) for uid, owed in calculated]

    if stype == "EXACT":
        exact_inputs = [
            (uid, Decimal(str(val)).quantize(Decimal("0.01")))
            for uid, val in participants
            if val is not None
        ]
        calculated = calculate_exact_splits(amount, exact_inputs)
        return [(uid, owed, "EXACT", owed) for uid, owed in calculated]

    if stype == "SHARES":
        merged_shares: dict[uuid.UUID, Decimal] = {}
        for uid, val in participants:
            s_val = Decimal(str(val if val is not None else 1)).quantize(Decimal("0.01"))
            merged_shares[uid] = merged_shares.get(uid, Decimal("0.00")) + s_val
        calculated = calculate_share_splits(amount, list(merged_shares.items()), payer_id=payer_id)
        return [(uid, owed, "SHARES", merged_shares[uid]) for uid, owed in calculated]

    # Default: EQUAL
    p_ids = [uid for uid, _ in participants]
    calculated = calculate_equal_splits(amount, p_ids, payer_id=payer_id)
    return [(uid, owed, "EQUAL", None) for uid, owed in calculated]


def recalculate_splits_on_amount_change(
    new_amount: Decimal,
    existing_splits: list[ExpenseSplit],
    payer_id: uuid.UUID,
) -> None:
    """
    Rebalance existing ExpenseSplit rows in-place when an expense's total amount is edited,
    preserving PERCENTAGE, SHARES, or EXACT proportions instead of overwriting with EQUAL.
    """
    if not existing_splits:
        return

    first_type = (existing_splits[0].split_type or "EQUAL").upper()
    if first_type == "PERCENTAGE" and all(s.split_value is not None for s in existing_splits):
        pct_inputs = [(s.user_id, Decimal(str(s.split_value))) for s in existing_splits]
        recalculated = dict(calculate_percentage_splits(new_amount, pct_inputs, payer_id=payer_id))
        for s in existing_splits:
            if s.user_id in recalculated:
                s.owed_amount = recalculated[s.user_id]
        return

    if first_type == "SHARES" and all(s.split_value is not None for s in existing_splits):
        share_inputs = [(s.user_id, Decimal(str(s.split_value))) for s in existing_splits]
        recalculated = dict(calculate_share_splits(new_amount, share_inputs, payer_id=payer_id))
        for s in existing_splits:
            if s.user_id in recalculated:
                s.owed_amount = recalculated[s.user_id]
        return

    if first_type == "EXACT" and sum(s.owed_amount for s in existing_splits) > Decimal("0.00"):
        ratio_inputs = [(s.user_id, s.owed_amount) for s in existing_splits]
        recalculated = dict(calculate_share_splits(new_amount, ratio_inputs, payer_id=payer_id))
        for s in existing_splits:
            if s.user_id in recalculated:
                s.owed_amount = recalculated[s.user_id]
                s.split_value = recalculated[s.user_id]
        return

    p_ids = [s.user_id for s in existing_splits]
    recalculated = dict(calculate_equal_splits(new_amount, p_ids, payer_id=payer_id))
    for s in existing_splits:
        if s.user_id in recalculated:
            s.owed_amount = recalculated[s.user_id]


async def get_user_balances(
    session: AsyncSession,
    user_id: uuid.UUID,
    group_id: uuid.UUID | None = None,
) -> BalanceSummary:
    """
    Compute net balances for a user across active shared expenses and settlements using SQL aggregation.

    Positive net_balance means others owe the user.
    Negative net_balance means the user owes others.
    """
    net_by_user: dict[uuid.UUID, Decimal] = defaultdict(lambda: Decimal("0.00"))

    # 1A. What others owe user_id (user_id paid, other user is in ExpenseSplit)
    owed_to_user_stmt = (
        select(
            ExpenseSplit.user_id,
            func.coalesce(func.sum(ExpenseSplit.owed_amount), Decimal("0.00")),
        )
        .join(Expense, Expense.id == ExpenseSplit.expense_id)
        .where(
            Expense.is_deleted.is_(False),
            Expense.is_personal.is_(False),
            Expense.payer_id == user_id,
            ExpenseSplit.user_id != user_id,
        )
        .group_by(ExpenseSplit.user_id)
    )
    if group_id:
        owed_to_user_stmt = owed_to_user_stmt.where(Expense.group_id == group_id)

    for other_id, total_owed in (await session.execute(owed_to_user_stmt)).all():
        net_by_user[other_id] += Decimal(str(total_owed)).quantize(Decimal("0.01"))

    # 1B. What user_id owes others (someone else paid, user_id is in ExpenseSplit)
    user_owes_stmt = (
        select(
            Expense.payer_id,
            func.coalesce(func.sum(ExpenseSplit.owed_amount), Decimal("0.00")),
        )
        .join(Expense, Expense.id == ExpenseSplit.expense_id)
        .where(
            Expense.is_deleted.is_(False),
            Expense.is_personal.is_(False),
            Expense.payer_id != user_id,
            ExpenseSplit.user_id == user_id,
        )
        .group_by(Expense.payer_id)
    )
    if group_id:
        user_owes_stmt = user_owes_stmt.where(Expense.group_id == group_id)

    for other_id, total_owed in (await session.execute(user_owes_stmt)).all():
        net_by_user[other_id] -= Decimal(str(total_owed)).quantize(Decimal("0.01"))

    # 2A. Settlements paid by user_id to others
    paid_settle_stmt = (
        select(
            Settlement.payee_id,
            func.coalesce(func.sum(Settlement.amount), Decimal("0.00")),
        )
        .where(
            Settlement.is_deleted.is_(False),
            Settlement.payer_id == user_id,
        )
        .group_by(Settlement.payee_id)
    )
    if group_id:
        paid_settle_stmt = paid_settle_stmt.where(Settlement.group_id == group_id)

    for payee_id, total_paid in (await session.execute(paid_settle_stmt)).all():
        net_by_user[payee_id] += Decimal(str(total_paid)).quantize(Decimal("0.01"))

    # 2B. Settlements received by user_id from others
    rcvd_settle_stmt = (
        select(
            Settlement.payer_id,
            func.coalesce(func.sum(Settlement.amount), Decimal("0.00")),
        )
        .where(
            Settlement.is_deleted.is_(False),
            Settlement.payee_id == user_id,
        )
        .group_by(Settlement.payer_id)
    )
    if group_id:
        rcvd_settle_stmt = rcvd_settle_stmt.where(Settlement.group_id == group_id)

    for payer_uid, total_rcvd in (await session.execute(rcvd_settle_stmt)).all():
        net_by_user[payer_uid] -= Decimal(str(total_rcvd)).quantize(Decimal("0.01"))

    # Fetch user names and custom nicknames for friends in balance map
    other_user_ids = list(net_by_user.keys())
    user_names: dict[uuid.UUID, str] = {}
    user_nicknames: dict[uuid.UUID, str | None] = {}
    if other_user_ids:
        users_result = await session.execute(select(User).where(User.id.in_(other_user_ids)))
        for u in users_result.scalars().all():
            user_names[u.id] = u.name

        ships_result = await session.execute(
            select(Friendship).where(
                Friendship.user_id == user_id,
                Friendship.friend_id.in_(other_user_ids),
            )
        )
        for s in ships_result.scalars().all():
            user_nicknames[s.friend_id] = s.nickname

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
                nickname=user_nicknames.get(other_id),
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


async def get_self_expenses_summary(
    session: AsyncSession,
    user: User,
    limit: int = 50,
    offset: int = 0,
    category: str | None = None,
    month: int | None = None,
    year: int | None = None,
) -> SelfExpensesSummary:
    """
    Compute strictly solo / personal spending summary and paginated expenses for user
    using SQL aggregation on `expenses` (where `is_personal == True`).
    """
    now_ist = datetime.now(IST)
    target_year = year or now_ist.year
    target_month = month or now_ist.month

    period_start_ist = datetime(target_year, target_month, 1, 0, 0, 0, tzinfo=IST)
    if target_month == 12:
        period_end_ist = datetime(target_year + 1, 1, 1, 0, 0, 0, tzinfo=IST)
    else:
        period_end_ist = datetime(target_year, target_month + 1, 1, 0, 0, 0, tzinfo=IST)

    period_start_utc = period_start_ist.astimezone(UTC)
    period_end_utc = period_end_ist.astimezone(UTC)

    base_filters = [
        Expense.is_deleted.is_(False),
        Expense.is_personal.is_(True),
        Expense.payer_id == user.id,
    ]

    # 1. All-time total solo spending
    total_stmt = select(func.coalesce(func.sum(Expense.amount), Decimal("0.00"))).where(*base_filters)
    total_spending = Decimal(str((await session.execute(total_stmt)).scalar_one())).quantize(Decimal("0.01"))

    # 2. Target month (or current IST month) spending
    month_stmt = select(func.coalesce(func.sum(Expense.amount), Decimal("0.00"))).where(
        *base_filters,
        Expense.expense_date >= period_start_utc,
        Expense.expense_date < period_end_utc,
    )
    this_month_spending = Decimal(str((await session.execute(month_stmt)).scalar_one())).quantize(Decimal("0.01"))

    # 3. Category breakdown (filtered by period if month/year was passed)
    cat_filters = list(base_filters)
    if month is not None or year is not None:
        cat_filters.extend(
            [
                Expense.expense_date >= period_start_utc,
                Expense.expense_date < period_end_utc,
            ]
        )

    cat_stmt = (
        select(
            Expense.category,
            func.coalesce(func.sum(Expense.amount), Decimal("0.00")),
            func.count(Expense.id),
        )
        .where(*cat_filters)
        .group_by(Expense.category)
    )
    cat_rows = (await session.execute(cat_stmt)).all()
    cat_totals: dict[str, Decimal] = defaultdict(lambda: Decimal("0.00"))
    cat_counts: dict[str, int] = defaultdict(int)
    for raw_name, c_sum, c_cnt in cat_rows:
        norm_cat = (raw_name or "general").strip().lower() or "general"
        cat_totals[norm_cat] += Decimal(str(c_sum)).quantize(Decimal("0.01"))
        cat_counts[norm_cat] += int(c_cnt)

    categories = [
        SelfCategoryStat(category=cat, total_amount=amt, count=cat_counts[cat])
        for cat, amt in cat_totals.items()
    ]
    categories.sort(key=lambda c: c.total_amount, reverse=True)

    # 4. Paginated list & filtered count
    list_filters = list(cat_filters)
    if category is not None:
        clean_cat = category.strip().lower()
        list_filters.append(func.lower(func.coalesce(Expense.category, "general")) == clean_cat)

    count_stmt = select(func.count(Expense.id)).where(*list_filters)
    total_count = int((await session.execute(count_stmt)).scalar_one())

    list_stmt = (
        select(Expense)
        .where(*list_filters)
        .order_by(Expense.expense_date.desc(), Expense.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    paginated = (await session.execute(list_stmt)).scalars().all()
    expense_reads = [SelfExpenseRead.model_validate(e) for e in paginated]

    return SelfExpensesSummary(
        total_spending=total_spending,
        this_month_spending=this_month_spending,
        total_count=total_count,
        currency="INR",
        categories=categories,
        expenses=expense_reads,
    )
