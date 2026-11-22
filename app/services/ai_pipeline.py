"""AI Pipeline: Simplified Direct-to-Ledger Voice Execution Engine."""

import json
import re
import uuid
import zoneinfo
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from google import genai
from google.genai import types
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models.expense import Expense, ExpenseSplit, SelfExpense
from app.models.group import Group, GroupMember
from app.models.settlement import Settlement
from app.models.user import Friendship, User
from app.prompts.stage1_parser import STAGE_1_SYSTEM_PROMPT
from app.prompts.stage2_matcher import STAGE_2_SYSTEM_PROMPT
from app.schemas.ai import (
    ExpenseScope,
    IntentType,
    SplitTypeHint,
    StageOneOutput,
    StageTwoResolution,
)
from app.schemas.ledger import ExpenseRead, SelfExpenseRead
from app.schemas.settlement import SettlementRead
from app.schemas.voice import VoiceIngestResponse
from app.services.ledger_service import (
    calculate_equal_splits,
    calculate_percentage_splits,
    get_user_balances,
)

# IST Timezone (+05:30)
IST = zoneinfo.ZoneInfo("Asia/Kolkata")
SELF_TOKENS = {"me", "myself", "i", "maine", "mera"}


def resolve_expense_date(expense_date_iso: str | None) -> datetime:
    """
    Parse AI-extracted expense_date_iso into a timezone-aware UTC datetime.

    - Date-only strings ('YYYY-MM-DD') for today (IST) resolve to the current
      timestamp so same-day expenses retain exact chronological order.
    - Date-only strings for past/future dates anchor at 12:00 IST (06:30 UTC)
      so calendar dates never drift across timezone boundaries.
    - Full ISO timestamps preserve their time (defaulting naive ones to IST).
    """
    now_utc = datetime.now(UTC)
    if not expense_date_iso or not expense_date_iso.strip():
        return now_utc

    raw = expense_date_iso.strip()
    now_ist = now_utc.astimezone(IST)

    if len(raw) == 10 and "T" not in raw and " " not in raw:
        try:
            parsed_date = date.fromisoformat(raw)
        except ValueError:
            return now_utc
        if parsed_date == now_ist.date():
            return now_utc
        return datetime(
            parsed_date.year,
            parsed_date.month,
            parsed_date.day,
            12,
            0,
            0,
            tzinfo=IST,
        ).astimezone(UTC)

    try:
        parsed_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return now_utc

    if parsed_dt.tzinfo is None:
        if (
            parsed_dt.hour == 0
            and parsed_dt.minute == 0
            and parsed_dt.second == 0
            and parsed_dt.microsecond == 0
        ):
            if parsed_dt.date() == now_ist.date():
                return now_utc
            return datetime(
                parsed_dt.year,
                parsed_dt.month,
                parsed_dt.day,
                12,
                0,
                0,
                tzinfo=IST,
            ).astimezone(UTC)
        parsed_dt = parsed_dt.replace(tzinfo=IST)

    return parsed_dt.astimezone(UTC)


def get_genai_client() -> genai.Client:
    """Instantiate official google-genai client with Vertex AI or API key."""
    if settings.GOOGLE_GENAI_USE_VERTEXAI:
        client_kwargs: dict[str, Any] = {
            "vertexai": True,
            "project": settings.GOOGLE_CLOUD_PROJECT,
            "location": settings.GOOGLE_CLOUD_LOCATION,
            "http_options": types.HttpOptions(timeout=60000),
        }
        if settings.GEMINI_API_KEY:
            client_kwargs["api_key"] = settings.GEMINI_API_KEY
        return genai.Client(**client_kwargs)

    return genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=60000),
    )


async def get_user_directory(
    session: AsyncSession,
    user: User,
) -> list[dict[str, Any]]:
    """Build directory of current user and friends with custom nicknames and aliases."""
    directory = [
        {
            "user_id": str(user.id),
            "name": user.name,
            "email": user.email,
            "nickname": "Me",
            "aliases": ["me", "myself", "i", "mera", "maine"],
            "is_caller": True,
        }
    ]

    query = select(Friendship).where(Friendship.user_id == user.id)
    friendships = (await session.execute(query)).scalars().all()

    friend_ids = [f.friend_id for f in friendships]
    if friend_ids:
        f_users = (await session.execute(select(User).where(User.id.in_(friend_ids)))).scalars().all()
        user_map = {u.id: u for u in f_users}

        for f in friendships:
            friend_user = user_map.get(f.friend_id)
            if friend_user:
                custom_aliases = [a.strip() for a in (f.aliases or []) if a and a.strip()]
                directory.append(
                    {
                        "user_id": str(friend_user.id),
                        "name": friend_user.name,
                        "email": friend_user.email,
                        "nickname": f.nickname,
                        "aliases": custom_aliases,
                        "is_caller": False,
                    }
                )

    return directory


async def get_user_groups_context(
    session: AsyncSession,
    user: User,
) -> list[dict[str, Any]]:
    """Fetch all groups the caller belongs to, including member names and IDs."""
    query = (
        select(Group)
        .join(GroupMember, Group.id == GroupMember.group_id)
        .where(GroupMember.user_id == user.id)
    )
    groups = (await session.execute(query)).scalars().all()

    groups_data = []
    for g in groups:
        mems_query = (
            select(GroupMember, User)
            .join(User, GroupMember.user_id == User.id)
            .where(GroupMember.group_id == g.id)
        )
        mem_rows = (await session.execute(mems_query)).all()
        members_info = [
            {"user_id": str(u.id), "name": u.name, "email": u.email}
            for _, u in mem_rows
        ]
        groups_data.append(
            {
                "group_id": str(g.id),
                "name": g.name,
                "aliases": [g.name.lower(), g.name.lower().replace(" group", "")],
                "members": members_info,
            }
        )

    return groups_data


def _score_group_match(raw_clean: str, group: dict[str, Any]) -> tuple[int, int] | None:
    """Score how well a spoken group name matches a user group."""
    g_name = str(group.get("name", "")).strip().lower()
    aliases = [str(a).strip().lower() for a in group.get("aliases", []) if a and str(a).strip()]
    candidates = ([g_name] if g_name else []) + aliases

    if raw_clean in candidates:
        return (1, -len(raw_clean))

    raw_word_pat = re.compile(rf"\b{re.escape(raw_clean)}\b")
    for cand in candidates:
        if raw_word_pat.search(cand) or re.search(rf"\b{re.escape(cand)}\b", raw_clean):
            return (2, -len(cand))

    if len(raw_clean) >= 3:
        for cand in candidates:
            if len(cand) >= 3 and (raw_clean in cand or cand in raw_clean):
                return (3, -len(cand))

    return None


def resolve_group_strict(
    group_raw: str | None,
    user_groups: list[dict[str, Any]],
) -> tuple[uuid.UUID | None, list[dict[str, Any]]]:
    """Resolve a spoken group name strictly, returning (resolved_id, matching_groups)."""
    if not group_raw or not group_raw.strip():
        return None, []
    raw_clean = group_raw.strip().lower()

    best_score: tuple[int, int] | None = None
    tied_groups: list[dict[str, Any]] = []

    for g in user_groups:
        score = _score_group_match(raw_clean, g)
        if score is None:
            continue
        if best_score is None or score < best_score:
            best_score = score
            tied_groups = [g]
        elif score == best_score:
            tied_groups.append(g)

    if len(tied_groups) == 1:
        return uuid.UUID(tied_groups[0]["group_id"]), tied_groups
    return None, tied_groups


def resolve_group_id(group_raw: str | None, user_groups: list[dict[str, Any]]) -> uuid.UUID | None:
    """Resolve raw spoken group name to an actual group ID when unambiguous."""
    resolved_id, _ = resolve_group_strict(group_raw, user_groups)
    return resolved_id


def _score_member_match(raw_clean: str, member: dict[str, Any]) -> tuple[int, int] | None:
    """
    Score how well a spoken raw name matches a directory member.

    Lower tier is higher priority; more negative length breaks ties in favor of
    more specific matches (e.g. 'rishabh sharma' over 'rishabh').
    """
    custom_labels: list[str] = []
    if member.get("nickname") and str(member["nickname"]).strip():
        custom_labels.append(str(member["nickname"]).strip().lower())
    for alias in member.get("aliases", []):
        if alias and str(alias).strip():
            custom_labels.append(str(alias).strip().lower())

    real_name = str(member.get("name", "")).strip().lower()

    if raw_clean in custom_labels:
        return (1, -len(raw_clean))

    if real_name and raw_clean == real_name:
        return (2, -len(raw_clean))

    raw_word_pat = re.compile(rf"\b{re.escape(raw_clean)}\b")
    for lbl in custom_labels:
        if raw_word_pat.search(lbl) or re.search(rf"\b{re.escape(lbl)}\b", raw_clean):
            return (3, -len(lbl))

    if real_name and (
        raw_word_pat.search(real_name) or re.search(rf"\b{re.escape(real_name)}\b", raw_clean)
    ):
        return (4, -len(real_name))

    if len(raw_clean) >= 3:
        for lbl in custom_labels:
            if len(lbl) >= 3 and (raw_clean in lbl or lbl in raw_clean):
                return (5, -len(lbl))
        if real_name and len(real_name) >= 3 and (raw_clean in real_name or real_name in raw_clean):
            return (5, -len(real_name))

    return None


def resolve_participants_strict(
    raw_names: list[str],
    user_directory: list[dict[str, Any]],
    caller_id: uuid.UUID,
    group_members: list[dict[str, Any]] | None = None,
) -> tuple[list[uuid.UUID], list[str], list[tuple[str, list[dict[str, Any]]]]]:
    """
    Strictly resolve spoken participant names without silent fallback to caller_id.

    Returns:
        (matched_ids, unmatched_names, ambiguous_matches)
    """
    combined_directory = list(user_directory)
    if group_members:
        existing_ids = {d["user_id"] for d in combined_directory}
        for gm in group_members:
            if gm["user_id"] not in existing_ids:
                combined_directory.append(
                    {
                        "user_id": gm["user_id"],
                        "name": gm["name"],
                        "email": gm["email"],
                        "nickname": None,
                        "aliases": [],
                        "is_caller": False,
                    }
                )

    matched_ids: set[uuid.UUID] = set()
    unmatched_names: list[str] = []
    ambiguous_matches: list[tuple[str, list[dict[str, Any]]]] = []

    for raw in raw_names:
        raw_clean = raw.strip().lower()
        if not raw_clean:
            continue
        if raw_clean in SELF_TOKENS:
            matched_ids.add(caller_id)
            continue
        if raw_clean in ["all", "sab", "everyone"] and group_members:
            matched_ids.add(caller_id)
            for gm in group_members:
                matched_ids.add(uuid.UUID(gm["user_id"]))
            continue

        best_score: tuple[int, int] | None = None
        tied_members: list[dict[str, Any]] = []

        for member in combined_directory:
            if member.get("is_caller"):
                continue
            score = _score_member_match(raw_clean, member)
            if score is None:
                continue
            if best_score is None or score < best_score:
                best_score = score
                tied_members = [member]
            elif score == best_score:
                tied_members.append(member)

        if len(tied_members) == 1:
            matched_ids.add(uuid.UUID(tied_members[0]["user_id"]))
        elif len(tied_members) >= 2:
            ambiguous_matches.append((raw.strip(), tied_members))
        else:
            unmatched_names.append(raw.strip())

    if matched_ids and caller_id not in matched_ids and any(
        r.strip().lower() in (SELF_TOKENS | {"all", "sab", "everyone"}) for r in raw_names
    ):
        matched_ids.add(caller_id)

    return list(matched_ids), unmatched_names, ambiguous_matches


def resolve_participant_ids(
    raw_names: list[str],
    user_directory: list[dict[str, Any]],
    caller_id: uuid.UUID,
    group_members: list[dict[str, Any]] | None = None,
) -> list[uuid.UUID]:
    """Map spoken raw participant names to verified User IDs."""
    matched_ids, _, ambiguous_matches = resolve_participants_strict(
        raw_names=raw_names,
        user_directory=user_directory,
        caller_id=caller_id,
        group_members=group_members,
    )
    resolved = set(matched_ids)
    for _, tied in ambiguous_matches:
        if tied:
            resolved.add(uuid.UUID(tied[0]["user_id"]))
    if not resolved:
        resolved.add(caller_id)
    return list(resolved)


async def search_verified_candidates(
    session: AsyncSession,
    user_id: uuid.UUID,
    keywords: list[str],
    target_date: datetime | None = None,
    window_days: int = 3,
) -> list[Expense]:
    """Search active shared expenses accessible to user_id."""
    caller_split_subquery = select(ExpenseSplit.id).where(
        ExpenseSplit.expense_id == Expense.id,
        ExpenseSplit.user_id == user_id,
    )
    caller_group_subquery = select(GroupMember.id).where(
        GroupMember.group_id == Expense.group_id,
        GroupMember.user_id == user_id,
    )
    verified_query = select(Expense).where(
        Expense.is_deleted.is_(False),
        or_(
            Expense.payer_id == user_id,
            caller_split_subquery.exists(),
            caller_group_subquery.exists(),
        ),
    )
    if target_date:
        if target_date.tzinfo is None:
            target_date = target_date.replace(tzinfo=UTC)
        verified_query = verified_query.where(
            Expense.expense_date >= target_date - timedelta(days=window_days),
            Expense.expense_date <= target_date + timedelta(days=window_days),
        )

    if keywords:
        filters = []
        for kw in keywords:
            filters.append(Expense.description.ilike(f"%{kw}%"))
            filters.append(Expense.category.ilike(f"%{kw}%"))
        verified_query = verified_query.where(or_(*filters))

    return list((await session.execute(verified_query)).scalars().all())


async def search_self_candidates(
    session: AsyncSession,
    user_id: uuid.UUID,
    keywords: list[str],
    target_date: datetime | None = None,
    window_days: int = 3,
) -> list[SelfExpense]:
    """Search active solo/self expenses owned by user_id."""
    self_query = select(SelfExpense).where(
        SelfExpense.is_deleted.is_(False),
        SelfExpense.user_id == user_id,
    )
    if target_date:
        if target_date.tzinfo is None:
            target_date = target_date.replace(tzinfo=UTC)
        self_query = self_query.where(
            SelfExpense.expense_date >= target_date - timedelta(days=window_days),
            SelfExpense.expense_date <= target_date + timedelta(days=window_days),
        )

    if keywords:
        filters = []
        for kw in keywords:
            filters.append(SelfExpense.description.ilike(f"%{kw}%"))
            filters.append(SelfExpense.category.ilike(f"%{kw}%"))
        self_query = self_query.where(or_(*filters))

    return list((await session.execute(self_query)).scalars().all())


async def search_db_candidates(
    session: AsyncSession,
    user_id: uuid.UUID,
    keywords: list[str],
    target_date: datetime | None = None,
    window_days: int = 3,
) -> tuple[list[SelfExpense], list[Expense]]:
    """Search candidate records across self_expenses and shared expenses."""
    self_cands = await search_self_candidates(session, user_id, keywords, target_date, window_days)
    shared_cands = await search_verified_candidates(session, user_id, keywords, target_date, window_days)
    return self_cands, shared_cands


def _voice_rejection(
    error_code: str,
    message: str,
    action_type: str = "CREATE",
    is_ambiguous: bool = False,
    status: str = "ERROR",
) -> VoiceIngestResponse:
    """Build a consistent non-mutating VoiceIngestResponse when confidence/resolution fails."""
    return VoiceIngestResponse(
        status=status,
        action_type=action_type,
        count=0,
        expenses=[],
        self_expenses=[],
        spoken_confirmation=message,
        clarification_prompt=message,
        error_code=error_code,
        is_ambiguous=is_ambiguous,
    )


def _format_candidate_label(member: dict[str, Any]) -> str:
    """Format a directory member with nickname if available for clarification messages."""
    nick = member.get("nickname")
    name = member.get("name", "Friend")
    if nick and nick.strip().lower() != name.strip().lower():
        return f"{nick} ({name})"
    return name


def _is_pure_self_item(item: Any, explicit_group_id: uuid.UUID | None) -> bool:
    """Determine if an extracted CREATE item strictly targets personal (SELF) spending."""
    if explicit_group_id is not None or (item.group_raw and item.group_raw.strip()):
        return False
    if item.percentage_splits:
        return False
    non_empty_parts = [p.strip().lower() for p in item.participants_raw if p and p.strip()]
    if any(p not in SELF_TOKENS for p in non_empty_parts):
        return False
    if item.scope == ExpenseScope.SELF:
        return True
    # Backward compatibility when Stage 1 omits scope and passes participants_raw=["me"]
    return bool(non_empty_parts) and all(p in SELF_TOKENS for p in non_empty_parts)


async def _handle_create_items(
    session: AsyncSession,
    user: User,
    stage1_output: StageOneOutput,
    user_directory: list[dict[str, Any]],
    user_groups: list[dict[str, Any]],
    explicit_group_id: uuid.UUID | None,
) -> VoiceIngestResponse:
    """
    Unified, deterministic handler for CREATE and MULTI_CREATE.

    Routes strictly personal items to `self_expenses` and shared friend/group items
    (equal or percentage splits, today or past dates) to `expenses` + `expense_splits`
    in one atomic transaction.
    """
    if not stage1_output.items:
        return _voice_rejection(
            "LOW_CONFIDENCE",
            "Couldn't catch the expense details. Please speak again clearly.",
            action_type="CREATE",
        )

    prepared_self: list[tuple[Any, Decimal, datetime]] = []
    prepared_shared: list[tuple[Any, uuid.UUID | None, Decimal, datetime, list[tuple[uuid.UUID, Decimal, str]]]] = []

    for item in stage1_output.items:
        amt_decimal = Decimal(str(item.amount))
        if amt_decimal <= Decimal("0.00"):
            return _voice_rejection(
                "INVALID_AMOUNT",
                "Couldn't catch a valid amount. Please speak again clearly with the amount.",
                action_type="CREATE",
            )

        if not item.description or not item.description.strip():
            return _voice_rejection(
                "LOW_CONFIDENCE",
                "Couldn't catch what the expense was for. Please speak again clearly.",
                action_type="CREATE",
            )

        item_date = resolve_expense_date(item.expense_date_iso)

        # 1. Personal / Solo path -> `self_expenses`
        if _is_pure_self_item(item, explicit_group_id):
            prepared_self.append((item, amt_decimal, item_date))
            continue

        # 2. Shared path -> `expenses` + `expense_splits`
        resolved_gid = explicit_group_id
        if not resolved_gid and item.group_raw:
            resolved_gid, matching_groups = resolve_group_strict(item.group_raw, user_groups)
            if len(matching_groups) >= 2:
                opts = ", ".join(g["name"] for g in matching_groups)
                return _voice_rejection(
                    "AMBIGUOUS_GROUP",
                    f"Found multiple groups matching '{item.group_raw}' ({opts}). Please speak again with the full group name.",
                    action_type="CREATE",
                    is_ambiguous=True,
                    status="AMBIGUOUS",
                )
            if not resolved_gid:
                return _voice_rejection(
                    "UNKNOWN_GROUP",
                    f"Couldn't find group '{item.group_raw}'. Please speak again with the exact group name.",
                    action_type="CREATE",
                )

        group_mems = None
        if resolved_gid:
            matched_g = next((g for g in user_groups if g["group_id"] == str(resolved_gid)), None)
            if matched_g:
                group_mems = matched_g["members"]

        prepared_splits: list[tuple[uuid.UUID, Decimal, str]] = []

        if item.split_type == SplitTypeHint.PERCENTAGE and item.percentage_splits:
            pct_tuples: list[tuple[uuid.UUID, Decimal]] = []
            for ps in item.percentage_splits:
                p_uids, unmatched_ps, ambiguous_ps = resolve_participants_strict(
                    [ps.participant_raw], user_directory, user.id, group_mems
                )
                if ambiguous_ps:
                    raw_name, tied_members = ambiguous_ps[0]
                    opts = ", ".join(_format_candidate_label(m) for m in tied_members)
                    return _voice_rejection(
                        "AMBIGUOUS_FRIEND",
                        f"Found multiple friends matching '{raw_name}' ({opts}). Please speak again using their nickname.",
                        action_type="CREATE",
                        is_ambiguous=True,
                        status="AMBIGUOUS",
                    )
                if unmatched_ps or not p_uids:
                    return _voice_rejection(
                        "UNKNOWN_FRIEND",
                        f"Couldn't find friend '{ps.participant_raw}'. Please speak again using their nickname or add them first.",
                        action_type="CREATE",
                    )
                pct_tuples.append((p_uids[0], Decimal(str(ps.percentage))))

            total_pct = sum(pct for _, pct in pct_tuples)
            if not pct_tuples or abs(total_pct - Decimal("100.00")) > Decimal("0.05"):
                return _voice_rejection(
                    "UNRESOLVED_SPLIT",
                    "The split percentages didn't add up to 100%. Please speak again clearly.",
                    action_type="CREATE",
                )

            for uid, owed in calculate_percentage_splits(amt_decimal, pct_tuples, payer_id=user.id):
                prepared_splits.append((uid, owed, "PERCENTAGE"))
        else:
            p_ids, unmatched_names, ambiguous_matches = resolve_participants_strict(
                item.participants_raw, user_directory, user.id, group_mems
            )
            if ambiguous_matches:
                raw_name, tied_members = ambiguous_matches[0]
                opts = ", ".join(_format_candidate_label(m) for m in tied_members)
                return _voice_rejection(
                    "AMBIGUOUS_FRIEND",
                    f"Found multiple friends matching '{raw_name}' ({opts}). Please speak again using their nickname.",
                    action_type="CREATE",
                    is_ambiguous=True,
                    status="AMBIGUOUS",
                )
            if unmatched_names:
                return _voice_rejection(
                    "UNKNOWN_FRIEND",
                    f"Couldn't find friend '{unmatched_names[0]}'. Please speak again using their nickname or add them first.",
                    action_type="CREATE",
                )

            if not p_ids:
                if group_mems:
                    p_ids = [uuid.UUID(m["user_id"]) for m in group_mems]
                else:
                    return _voice_rejection(
                        "LOW_CONFIDENCE",
                        "I wasn't sure who to add this expense for. Please speak again clearly.",
                        action_type="CREATE",
                    )

            for uid, owed in calculate_equal_splits(amt_decimal, p_ids, payer_id=user.id):
                prepared_splits.append((uid, owed, "EQUAL"))

        prepared_shared.append((item, resolved_gid, amt_decimal, item_date, prepared_splits))

    # All items validated cleanly -> commit atomically
    created_self_ids: list[uuid.UUID] = []
    created_shared_ids: list[uuid.UUID] = []

    for item, amt_decimal, item_date in prepared_self:
        self_exp = SelfExpense(
            user_id=user.id,
            description=item.description.strip(),
            amount=amt_decimal,
            currency=item.currency,
            category=item.category or "general",
            expense_date=item_date,
            is_deleted=False,
        )
        session.add(self_exp)
        await session.flush()
        created_self_ids.append(self_exp.id)

    for item, resolved_gid, amt_decimal, item_date, splits in prepared_shared:
        exp = Expense(
            group_id=resolved_gid,
            description=item.description.strip(),
            amount=amt_decimal,
            currency=item.currency,
            payer_id=user.id,
            category=item.category or "general",
            expense_date=item_date,
            is_deleted=False,
        )
        session.add(exp)
        await session.flush()

        for uid, owed, stype in splits:
            session.add(
                ExpenseSplit(
                    expense_id=exp.id,
                    user_id=uid,
                    owed_amount=owed,
                    split_type=stype,
                )
            )
        created_shared_ids.append(exp.id)

    await session.commit()

    created_self: list[SelfExpenseRead] = []
    for sid in created_self_ids:
        saved_s = (await session.execute(select(SelfExpense).where(SelfExpense.id == sid))).scalars().first()
        if saved_s:
            created_self.append(SelfExpenseRead.model_validate(saved_s))

    created_shared: list[ExpenseRead] = []
    for eid in created_shared_ids:
        saved_e = (
            await session.execute(
                select(Expense).options(selectinload(Expense.splits)).where(Expense.id == eid)
            )
        ).scalars().first()
        if saved_e:
            model = ExpenseRead.model_validate(saved_e)
            model.payer_name = user.name
            created_shared.append(model)

    total_count = len(created_self) + len(created_shared)
    return VoiceIngestResponse(
        status="SUCCESS",
        action_type="CREATE",
        count=total_count,
        expenses=created_shared,
        self_expenses=created_self,
        spoken_confirmation=stage1_output.spoken_confirmation
        or f"Done! Added {total_count} expense(s) to your ledger.",
        is_ambiguous=False,
    )


async def _handle_settle(
    session: AsyncSession,
    user: User,
    stage1_output: StageOneOutput,
    user_directory: list[dict[str, Any]],
    user_groups: list[dict[str, Any]],
    explicit_group_id: uuid.UUID | None,
) -> VoiceIngestResponse:
    """Deterministic handler for voice SETTLE commands."""
    s_item = stage1_output.settle_item
    if not s_item:
        return _voice_rejection(
            "LOW_CONFIDENCE",
            "Couldn't catch settlement details. Please speak again clearly.",
            action_type="SETTLE",
        )

    settle_amt = Decimal(str(s_item.amount))
    if settle_amt <= Decimal("0.00"):
        return _voice_rejection(
            "INVALID_AMOUNT",
            "Couldn't catch a valid settlement amount. Please speak again clearly.",
            action_type="SETTLE",
        )

    resolved_gid = explicit_group_id
    if not resolved_gid and s_item.group_raw:
        resolved_gid, matching_groups = resolve_group_strict(s_item.group_raw, user_groups)
        if len(matching_groups) >= 2:
            opts = ", ".join(g["name"] for g in matching_groups)
            return _voice_rejection(
                "AMBIGUOUS_GROUP",
                f"Found multiple groups matching '{s_item.group_raw}' ({opts}). Please speak again with the full group name.",
                action_type="SETTLE",
                is_ambiguous=True,
                status="AMBIGUOUS",
            )
        if not resolved_gid:
            return _voice_rejection(
                "UNKNOWN_GROUP",
                f"Couldn't find group '{s_item.group_raw}'. Please speak again with the exact group name.",
                action_type="SETTLE",
            )

    payee_ids, unmatched_payees, ambiguous_payees = resolve_participants_strict(
        [s_item.payee_raw], user_directory, user.id
    )
    if ambiguous_payees:
        raw_name, tied_members = ambiguous_payees[0]
        opts = ", ".join(_format_candidate_label(m) for m in tied_members)
        return _voice_rejection(
            "AMBIGUOUS_FRIEND",
            f"Found multiple friends matching '{raw_name}' ({opts}). Please speak again using their nickname.",
            action_type="SETTLE",
            is_ambiguous=True,
            status="AMBIGUOUS",
        )
    target_payee_id = next((pid for pid in payee_ids if pid != user.id), None)
    if unmatched_payees or not target_payee_id:
        return _voice_rejection(
            "UNKNOWN_FRIEND",
            f"Couldn't find friend '{s_item.payee_raw}' to settle with. Please speak again using their nickname.",
            action_type="SETTLE",
        )

    settlement = Settlement(
        payer_id=user.id,
        payee_id=target_payee_id,
        amount=settle_amt,
        currency=s_item.currency,
        group_id=resolved_gid,
        payment_method=s_item.payment_method or "UPI",
        notes=s_item.notes or "Voice settle up",
        settlement_date=datetime.now(UTC),
        is_deleted=False,
    )
    session.add(settlement)
    await session.commit()
    await session.refresh(settlement)

    payee_user = (await session.execute(select(User).where(User.id == target_payee_id))).scalars().first()
    payee_name = payee_user.name if payee_user else s_item.payee_raw
    conf = stage1_output.spoken_confirmation or f"Done! Settle up of ₹{settle_amt} paid to {payee_name} recorded."

    return VoiceIngestResponse(
        status="SUCCESS",
        action_type="SETTLE",
        count=1,
        settlement=SettlementRead(
            id=settlement.id,
            payer_id=settlement.payer_id,
            payer_name=user.name,
            payee_id=settlement.payee_id,
            payee_name=payee_name,
            amount=settlement.amount,
            currency=settlement.currency,
            group_id=settlement.group_id,
            payment_method=settlement.payment_method,
            transaction_reference=settlement.transaction_reference,
            notes=settlement.notes,
            settlement_date=settlement.settlement_date,
            is_deleted=settlement.is_deleted,
            created_at=settlement.created_at,
        ),
        spoken_confirmation=conf,
        is_ambiguous=False,
    )


async def process_voice_pipeline(
    session: AsyncSession,
    user: User,
    command_text: str,
    client_timestamp_iso: str | None = None,
    group_id: uuid.UUID | None = None,
) -> VoiceIngestResponse:
    """
    Simplified Voice Processing Pipeline:
    1. Stage 1 (gemini-3.8-flash): Extract intent, scope (SELF/SHARED), amounts, entities, dates.
    2. Deterministic Execution in Python for CREATE, MULTI_CREATE, SETTLE, and GET:
       - SELF -> `self_expenses` table
       - SHARED -> `expenses` + `expense_splits`
    3. Stage 2 (gemini-4-hydrogen): Invoked ONLY for UPDATE and DELETE candidate disambiguation.
    """
    client = get_genai_client()

    user_directory = await get_user_directory(session, user)
    user_groups = await get_user_groups_context(session, user)
    user_balances = await get_user_balances(session, user.id, group_id=group_id)

    now_ist = client_timestamp_iso or datetime.now(IST).isoformat()
    directory_str = json.dumps(
        [
            {
                "name": d["name"],
                "nickname": d["nickname"],
                "aliases": d["aliases"],
                "is_caller": d["is_caller"],
            }
            for d in user_directory
        ],
        ensure_ascii=False,
    )
    groups_str = json.dumps(
        [{"name": g["name"], "members": [m["name"] for m in g["members"]]} for g in user_groups],
        ensure_ascii=False,
    )

    stage1_prompt = (
        f"Current User: {user.name} (ID: {user.id})\n"
        f"Client Timestamp (IST): {now_ist}\n"
        f"Active Group Filter ID: {group_id}\n"
        f"Known Friends & Directory: {directory_str}\n"
        f"Known Groups: {groups_str}\n"
        f"Voice Command: {command_text}"
    )

    stage1_response = await client.aio.models.generate_content(
        model=settings.STAGE1_MODEL,
        contents=stage1_prompt,
        config=types.GenerateContentConfig(
            system_instruction=STAGE_1_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=StageOneOutput,
            temperature=0.1,
        ),
    )

    stage1_output = StageOneOutput.model_validate_json(stage1_response.text)

    # Gate 1: Confidence & Intent check
    stage1_conf = (stage1_output.confidence or "HIGH").strip().upper()
    if stage1_output.intent == IntentType.UNKNOWN:
        msg = (
            stage1_output.spoken_confirmation
            or "Sound was not clear or I couldn't understand that. Please speak again clearly."
        )
        return _voice_rejection("LOW_CONFIDENCE", msg, action_type="UNKNOWN")

    if stage1_conf != "HIGH":
        msg = (
            stage1_output.spoken_confirmation
            or "I wasn't completely sure what you said. Please speak again clearly."
        )
        return _voice_rejection("LOW_CONFIDENCE", msg, action_type=str(stage1_output.intent))

    # Deterministic Single Path: CREATE / MULTI_CREATE
    if stage1_output.intent in (IntentType.CREATE, IntentType.MULTI_CREATE):
        return await _handle_create_items(
            session=session,
            user=user,
            stage1_output=stage1_output,
            user_directory=user_directory,
            user_groups=user_groups,
            explicit_group_id=group_id,
        )

    # Deterministic Single Path: SETTLE
    if stage1_output.intent == IntentType.SETTLE:
        return await _handle_settle(
            session=session,
            user=user,
            stage1_output=stage1_output,
            user_directory=user_directory,
            user_groups=user_groups,
            explicit_group_id=group_id,
        )

    # Deterministic Single Path: GET
    if stage1_output.intent == IntentType.GET:
        conf = stage1_output.spoken_confirmation
        if not conf:
            if user_balances.balances:
                first = user_balances.balances[0]
                if first.net_balance > 0:
                    conf = f"{first.user_name} owes you ₹{first.net_balance:.2f}."
                else:
                    conf = f"You owe {first.user_name} ₹{abs(first.net_balance):.2f}."
            else:
                conf = "You are all settled up with everyone!"

        return VoiceIngestResponse(
            status="SUCCESS",
            action_type="GET",
            count=len(user_balances.balances),
            balance_summary=user_balances,
            spoken_confirmation=conf,
            is_ambiguous=False,
        )

    # -------------------------------------------------------------
    # STAGE 2 PATH: Candidate Disambiguation for UPDATE & DELETE only
    # -------------------------------------------------------------
    search_keywords: list[str] = []
    target_dt: datetime | None = None

    if stage1_output.search_query:
        search_keywords = stage1_output.search_query.description_keywords
        if stage1_output.search_query.date_target_iso:
            try:
                target_dt = datetime.fromisoformat(stage1_output.search_query.date_target_iso)
                if target_dt.tzinfo is None:
                    target_dt = target_dt.replace(tzinfo=UTC)
            except ValueError:
                pass
    elif stage1_output.items:
        search_keywords = [stage1_output.items[0].description]

    self_cands, shared_cands = await search_db_candidates(
        session=session,
        user_id=user.id,
        keywords=search_keywords,
        target_date=target_dt,
    )

    stage2_context = {
        "stage1_extraction": stage1_output.model_dump(),
        "candidate_shared_expenses": [
            {
                "id": str(e.id),
                "source": "shared",
                "description": e.description,
                "amount": float(e.amount),
                "date": e.expense_date.isoformat(),
                "category": e.category,
                "payer_id": str(e.payer_id),
            }
            for e in shared_cands
        ],
        "candidate_self_expenses": [
            {
                "id": str(se.id),
                "source": "self",
                "description": se.description,
                "amount": float(se.amount),
                "date": se.expense_date.isoformat(),
                "category": se.category,
            }
            for se in self_cands
        ],
    }

    stage2_response = await client.aio.models.generate_content(
        model=settings.STAGE2_MODEL,
        contents=json.dumps(stage2_context),
        config=types.GenerateContentConfig(
            system_instruction=STAGE_2_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=StageTwoResolution,
            temperature=0.1,
        ),
    )

    stage2_output = StageTwoResolution.model_validate_json(stage2_response.text)
    stage2_conf = (stage2_output.confidence or "HIGH").strip().upper()

    if stage2_conf == "AMBIGUOUS" or len(stage2_output.ambiguous_candidate_ids) >= 2:
        msg = (
            stage2_output.clarification_question
            or "Found multiple matching records. Please specify which one you mean."
        )
        return VoiceIngestResponse(
            status="AMBIGUOUS",
            action_type=stage2_output.action_type,
            count=0,
            spoken_confirmation=stage2_output.spoken_confirmation or msg,
            is_ambiguous=True,
            clarification_prompt=msg,
            error_code="AMBIGUOUS_CANDIDATE",
        )

    if stage2_conf != "HIGH" or stage2_output.action_type == "CLARIFICATION_REQUIRED":
        msg = (
            stage2_output.clarification_question
            or stage2_output.spoken_confirmation
            or "Sound or details were not clear enough. Please speak again clearly."
        )
        return _voice_rejection("LOW_CONFIDENCE", msg, action_type=stage2_output.action_type)

    if stage2_output.target_record_id:
        try:
            target_id = uuid.UUID(stage2_output.target_record_id)
        except ValueError:
            return _voice_rejection(
                "LOW_CONFIDENCE",
                "Couldn't identify which expense to modify. Please speak again clearly.",
                action_type=stage2_output.action_type,
            )

        shared_by_id = {e.id: e for e in shared_cands}
        self_by_id = {se.id: se for se in self_cands}

        if stage2_output.action_type == "UPDATE":
            if target_id in self_by_id or (
                stage2_output.target_source == "self" and target_id in self_by_id
            ):
                target_self = (
                    await session.execute(
                        select(SelfExpense).where(
                            SelfExpense.id == target_id,
                            SelfExpense.user_id == user.id,
                            SelfExpense.is_deleted.is_(False),
                        )
                    )
                ).scalars().first()
                if target_self:
                    if stage2_output.description:
                        target_self.description = stage2_output.description
                    if stage2_output.amount > 0:
                        target_self.amount = stage2_output.amount
                    if stage2_output.category:
                        target_self.category = stage2_output.category
                    target_self.updated_at = datetime.now(UTC)
                    await session.commit()
                    await session.refresh(target_self)
                    return VoiceIngestResponse(
                        status="SUCCESS",
                        action_type="UPDATE",
                        count=1,
                        self_expenses=[SelfExpenseRead.model_validate(target_self)],
                        spoken_confirmation=stage2_output.spoken_confirmation
                        or f"Done! Updated {target_self.description} to ₹{target_self.amount}.",
                        is_ambiguous=False,
                    )

            if target_id in shared_by_id:
                target_exp = (
                    await session.execute(
                        select(Expense)
                        .options(selectinload(Expense.splits))
                        .where(Expense.id == target_id, Expense.is_deleted.is_(False))
                    )
                ).scalars().first()
                if target_exp:
                    if stage2_output.description:
                        target_exp.description = stage2_output.description
                    if stage2_output.amount > 0:
                        target_exp.amount = stage2_output.amount
                    if stage2_output.category:
                        target_exp.category = stage2_output.category
                    target_exp.updated_at = datetime.now(UTC)

                    if target_exp.splits:
                        p_ids = [s.user_id for s in target_exp.splits]
                        recalculated = calculate_equal_splits(
                            amount=target_exp.amount,
                            participant_ids=p_ids,
                            payer_id=target_exp.payer_id,
                        )
                        for split in target_exp.splits:
                            for u_id, new_owed in recalculated:
                                if split.user_id == u_id:
                                    split.owed_amount = new_owed
                                    break

                    await session.commit()
                    await session.refresh(target_exp)
                    model = ExpenseRead.model_validate(target_exp)
                    model.payer_name = user.name if target_exp.payer_id == user.id else None
                    return VoiceIngestResponse(
                        status="SUCCESS",
                        action_type="UPDATE",
                        count=1,
                        expenses=[model],
                        spoken_confirmation=stage2_output.spoken_confirmation
                        or f"Done! Updated {target_exp.description} to ₹{target_exp.amount}.",
                        is_ambiguous=False,
                    )

        if stage2_output.action_type == "DELETE":
            if target_id in self_by_id:
                target_self = (
                    await session.execute(
                        select(SelfExpense).where(
                            SelfExpense.id == target_id,
                            SelfExpense.user_id == user.id,
                            SelfExpense.is_deleted.is_(False),
                        )
                    )
                ).scalars().first()
                if target_self:
                    target_self.is_deleted = True
                    target_self.updated_at = datetime.now(UTC)
                    await session.commit()
                    return VoiceIngestResponse(
                        status="SUCCESS",
                        action_type="DELETE",
                        count=1,
                        spoken_confirmation=stage2_output.spoken_confirmation
                        or f"Done! Deleted {target_self.description}.",
                        is_ambiguous=False,
                    )

            if target_id in shared_by_id:
                target_exp = (
                    await session.execute(
                        select(Expense).where(Expense.id == target_id, Expense.is_deleted.is_(False))
                    )
                ).scalars().first()
                if target_exp:
                    target_exp.is_deleted = True
                    target_exp.updated_at = datetime.now(UTC)
                    await session.commit()
                    return VoiceIngestResponse(
                        status="SUCCESS",
                        action_type="DELETE",
                        count=1,
                        spoken_confirmation=stage2_output.spoken_confirmation
                        or f"Done! Deleted {target_exp.description}.",
                        is_ambiguous=False,
                    )

    return _voice_rejection(
        "LOW_CONFIDENCE",
        stage2_output.clarification_question
        or stage2_output.spoken_confirmation
        or "Couldn't find that expense to modify. Please speak again clearly.",
        action_type=stage2_output.action_type,
    )
