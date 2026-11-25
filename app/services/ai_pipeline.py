"""AI Pipeline: Single-Pass Multimodal Direct-to-Ledger Voice Execution Engine."""

import json
import re
import uuid
import zoneinfo
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from google import genai
from google.genai import types
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models.expense import Expense, ExpenseSplit
from app.models.group import Group, GroupMember
from app.models.settlement import Settlement
from app.models.user import Friendship, User
from app.prompts.stage1_parser import STAGE_1_SYSTEM_PROMPT
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
    compute_splits,
    get_user_balances,
    recalculate_splits_on_amount_change,
)

IST = zoneinfo.ZoneInfo("Asia/Kolkata")
SELF_TOKENS = {"me", "myself", "i", "maine", "mera"}
ALL_GROUP_TOKENS = {"all", "sab", "everyone", "group"}


def resolve_expense_date(expense_date_iso: str | None) -> datetime:
    """
    Parse AI-extracted expense_date_iso into a timezone-aware UTC datetime.

    - Date-only strings ('YYYY-MM-DD') for today (IST) resolve to the current
      timestamp so same-day expenses retain exact chronological order.
    - Date-only strings for past/future dates anchor at 12:00 IST (06:30 UTC)
      so calendar dates never drift across timezone boundaries.
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
    """Build directory of current user and friends with custom nicknames and aliases in 1 JOIN query."""
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

    rows = (
        await session.execute(
            select(Friendship, User)
            .join(User, Friendship.friend_id == User.id)
            .where(Friendship.user_id == user.id)
        )
    ).all()

    for f, friend_user in rows:
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
    """Fetch all groups the caller belongs to and their members in 2 batch queries (no N+1 loop)."""
    groups = (
        await session.execute(
            select(Group)
            .join(GroupMember, Group.id == GroupMember.group_id)
            .where(GroupMember.user_id == user.id)
        )
    ).scalars().all()

    if not groups:
        return []

    group_ids = [g.id for g in groups]
    mem_rows = (
        await session.execute(
            select(GroupMember.group_id, User)
            .join(User, GroupMember.user_id == User.id)
            .where(GroupMember.group_id.in_(group_ids))
        )
    ).all()

    members_by_group: dict[uuid.UUID, list[dict[str, str]]] = {gid: [] for gid in group_ids}
    for gid, u in mem_rows:
        members_by_group[gid].append({"user_id": str(u.id), "name": u.name, "email": u.email})

    return [
        {
            "group_id": str(g.id),
            "name": g.name,
            "aliases": [g.name.lower(), g.name.lower().replace(" group", "")],
            "members": members_by_group.get(g.id, []),
        }
        for g in groups
    ]


def _has_word_prefix_match(raw_clean: str, candidate: str, min_len: int = 4) -> bool:
    """Check if any word token (>=4 chars) is a prefix of a candidate word token (avoids 3-char substring hijacking)."""
    if len(raw_clean) < min_len or len(candidate) < min_len:
        return False
    raw_tokens = [t for t in re.split(r"\W+", raw_clean) if len(t) >= min_len]
    cand_tokens = [t for t in re.split(r"\W+", candidate) if len(t) >= min_len]
    for rt in raw_tokens:
        for ct in cand_tokens:
            if ct.startswith(rt) or rt.startswith(ct):
                return True
    return False


def _score_group_match(raw_clean: str, group: dict[str, Any]) -> tuple[int, int] | None:
    """Score how well a spoken group name or UUID matches a user group."""
    if raw_clean == str(group.get("group_id", "")).strip().lower():
        return (0, -len(raw_clean))

    g_name = str(group.get("name", "")).strip().lower()
    aliases = [str(a).strip().lower() for a in group.get("aliases", []) if a and str(a).strip()]
    candidates = ([g_name] if g_name else []) + aliases

    if raw_clean in candidates:
        return (1, -len(raw_clean))

    raw_word_pat = re.compile(rf"\b{re.escape(raw_clean)}\b")
    for cand in candidates:
        if raw_word_pat.search(cand) or re.search(rf"\b{re.escape(cand)}\b", raw_clean):
            return (2, -len(cand))

    for cand in candidates:
        if _has_word_prefix_match(raw_clean, cand):
            return (3, -len(cand))

    return None


def resolve_group_strict(
    group_raw: str | None,
    user_groups: list[dict[str, Any]],
) -> tuple[uuid.UUID | None, list[dict[str, Any]]]:
    """Resolve a spoken group name or UUID strictly, returning (resolved_id, matching_groups)."""
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
    Score how well a spoken raw name or UUID matches a directory member.

    Tier 0: Exact UUID match
    Tier 1: Exact nickname or alias match
    Tier 2: Exact full name match
    Tier 3: Word-boundary match on nickname/alias
    Tier 4: Word-boundary match on real name
    """
    if raw_clean == str(member.get("user_id", "")).strip().lower():
        return (0, -len(raw_clean))

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

    for lbl in custom_labels:
        if _has_word_prefix_match(raw_clean, lbl):
            return (5, -len(lbl))
    if real_name and _has_word_prefix_match(raw_clean, real_name):
        return (5, -len(real_name))

    return None


def resolve_participants_strict(
    raw_names: list[str],
    user_directory: list[dict[str, Any]],
    caller_id: uuid.UUID,
    group_members: list[dict[str, Any]] | None = None,
) -> tuple[list[uuid.UUID], list[str], list[tuple[str, list[dict[str, Any]]]]]:
    """
    Strictly resolve spoken participant names or UUIDs without silent fallback to caller_id.

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

    matched_ids: list[uuid.UUID] = []
    unmatched_names: list[str] = []
    ambiguous_matches: list[tuple[str, list[dict[str, Any]]]] = []

    def _add_unique(uid: uuid.UUID) -> None:
        if uid not in matched_ids:
            matched_ids.append(uid)

    for raw in raw_names:
        raw_clean = raw.strip().lower()
        if not raw_clean:
            continue
        if raw_clean in SELF_TOKENS or raw_clean == str(caller_id).lower():
            _add_unique(caller_id)
            continue
        if raw_clean in ALL_GROUP_TOKENS and group_members:
            _add_unique(caller_id)
            for gm in group_members:
                _add_unique(uuid.UUID(gm["user_id"]))
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
            _add_unique(uuid.UUID(tied_members[0]["user_id"]))
        elif len(tied_members) >= 2:
            ambiguous_matches.append((raw.strip(), tied_members))
        else:
            unmatched_names.append(raw.strip())

    return matched_ids, unmatched_names, ambiguous_matches


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
    resolved = list(matched_ids)
    for _, tied in ambiguous_matches:
        if tied:
            uid = uuid.UUID(tied[0]["user_id"])
            if uid not in resolved:
                resolved.append(uid)
    if not resolved:
        resolved.append(caller_id)
    return resolved


async def search_db_candidates(
    session: AsyncSession,
    user_id: uuid.UUID,
    keywords: list[str],
    target_date: datetime | None = None,
    window_days: int = 3,
    amount_hint: float | None = None,
) -> tuple[list[Expense], list[Expense]]:
    """
    Search candidate records in unified `expenses` table, returning (self_cands, shared_cands).
    """
    caller_split_subquery = select(ExpenseSplit.id).where(
        ExpenseSplit.expense_id == Expense.id,
        ExpenseSplit.user_id == user_id,
    )
    caller_group_subquery = select(GroupMember.id).where(
        GroupMember.group_id == Expense.group_id,
        GroupMember.user_id == user_id,
    )

    base_query = (
        select(Expense)
        .options(selectinload(Expense.splits))
        .where(
            Expense.is_deleted.is_(False),
            or_(
                Expense.payer_id == user_id,
                caller_split_subquery.exists(),
                caller_group_subquery.exists(),
            ),
        )
        .order_by(Expense.expense_date.desc(), Expense.created_at.desc())
    )

    if target_date:
        if target_date.tzinfo is None:
            target_date = target_date.replace(tzinfo=UTC)
        base_query = base_query.where(
            Expense.expense_date >= target_date - timedelta(days=window_days),
            Expense.expense_date <= target_date + timedelta(days=window_days),
        )

    if keywords:
        # Split multi-word keywords into individual tokens so "petrol bill" matches "Petrol"
        tokens: list[str] = []
        stop_words = {"bill", "expense", "rupees", "rs", "inr", "wali", "wala"}
        for kw in keywords:
            for part in kw.strip().split():
                clean_part = part.strip().lower()
                if len(clean_part) >= 2 and clean_part not in stop_words:
                    tokens.append(clean_part)
        if not tokens:
            tokens = [kw.strip() for kw in keywords if kw.strip()]

        filters = []
        for tok in tokens:
            filters.append(Expense.description.ilike(f"%{tok}%"))
            filters.append(Expense.category.ilike(f"%{tok}%"))
        if filters:
            base_query = base_query.where(or_(*filters))

    all_cands = list((await session.execute(base_query)).scalars().all())

    # If amount_hint matches a subset of candidates, prioritize those
    if amount_hint is not None and amount_hint > 0 and len(all_cands) > 1:
        hint_dec = Decimal(str(amount_hint)).quantize(Decimal("0.01"))
        exact_amt_matches = [c for c in all_cands if c.amount == hint_dec]
        if exact_amt_matches:
            all_cands = exact_amt_matches

    self_cands = [c for c in all_cands if c.is_personal]
    shared_cands = [c for c in all_cands if not c.is_personal]
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
    if explicit_group_id is not None or (item.group_raw and item.group_raw.strip()) or item.group_id:
        return False
    if item.percentage_splits or item.exact_splits or item.share_splits:
        return False
    if item.payer_raw and item.payer_raw.strip().lower() not in SELF_TOKENS:
        return False
    non_empty_parts = [p.strip().lower() for p in item.participants_raw if p and p.strip()]
    if any(p not in SELF_TOKENS for p in non_empty_parts):
        return False
    if item.scope == ExpenseScope.SELF:
        return True
    return bool(non_empty_parts) and all(p in SELF_TOKENS for p in non_empty_parts)


async def _handle_create_items(
    session: AsyncSession,
    user: User,
    stage1_output: StageOneOutput,
    user_directory: list[dict[str, Any]],
    user_groups: list[dict[str, Any]],
    explicit_group_id: uuid.UUID | None,
    command_text: str | None = None,
) -> VoiceIngestResponse:
    """
    Deterministic handler for CREATE and MULTI_CREATE:
    - Resolves payer_raw (caller OR friend).
    - Supports EQUAL, PERCENTAGE, EXACT, and SHARES splits via unified compute_splits().
    - Defaults group expenses to all group members when no specific subset is named.
    - Writes personal (`is_personal=True`) and shared (`is_personal=False`) expenses atomically.
    """
    if not stage1_output.items:
        return _voice_rejection(
            "LOW_CONFIDENCE",
            "Couldn't catch the expense details. Please speak again clearly.",
            action_type="CREATE",
        )

    prepared_self: list[tuple[Any, Decimal, datetime]] = []
    prepared_shared: list[
        tuple[Any, uuid.UUID | None, uuid.UUID, Decimal, datetime, list[tuple[uuid.UUID, Decimal, str, Decimal | None]]]
    ] = []

    for item in stage1_output.items:
        amt_decimal = Decimal(str(item.amount)).quantize(Decimal("0.01"))
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

        # 1. Personal / Solo path -> Expense(is_personal=True)
        if _is_pure_self_item(item, explicit_group_id):
            prepared_self.append((item, amt_decimal, item_date))
            continue

        # 2. Shared path -> Expense(is_personal=False) + ExpenseSplit
        resolved_gid = explicit_group_id
        group_lookup_raw = item.group_id or item.group_raw
        if not resolved_gid and group_lookup_raw:
            resolved_gid, matching_groups = resolve_group_strict(group_lookup_raw, user_groups)
            if len(matching_groups) >= 2:
                opts = ", ".join(g["name"] for g in matching_groups)
                return _voice_rejection(
                    "AMBIGUOUS_GROUP",
                    f"Found multiple groups matching '{group_lookup_raw}' ({opts}). Please speak again with the full group name.",
                    action_type="CREATE",
                    is_ambiguous=True,
                    status="AMBIGUOUS",
                )
            if not resolved_gid:
                return _voice_rejection(
                    "UNKNOWN_GROUP",
                    f"Couldn't find group '{group_lookup_raw}'. Please speak again with the exact group name.",
                    action_type="CREATE",
                )

        group_mems = None
        if resolved_gid:
            matched_g = next((g for g in user_groups if g["group_id"] == str(resolved_gid)), None)
            if matched_g:
                group_mems = matched_g["members"]

        # Resolve Payer (supports both caller and a friend paying!)
        actual_payer_id = user.id
        payer_lookup = item.payer_user_id or item.payer_raw
        if payer_lookup and payer_lookup.strip().lower() not in SELF_TOKENS:
            payer_uids, unmatched_payer, ambiguous_payer = resolve_participants_strict(
                [payer_lookup], user_directory, user.id, group_mems
            )
            if ambiguous_payer:
                raw_name, tied_members = ambiguous_payer[0]
                opts = ", ".join(_format_candidate_label(m) for m in tied_members)
                return _voice_rejection(
                    "AMBIGUOUS_FRIEND",
                    f"Found multiple friends matching payer '{raw_name}' ({opts}). Please speak again using their nickname.",
                    action_type="CREATE",
                    is_ambiguous=True,
                    status="AMBIGUOUS",
                )
            if unmatched_payer or not payer_uids:
                return _voice_rejection(
                    "UNKNOWN_FRIEND",
                    f"Couldn't find friend '{payer_lookup}' who paid. Please speak again using their nickname.",
                    action_type="CREATE",
                )
            actual_payer_id = payer_uids[0]

        # Resolve Participants & Compute Splits via unified SplitEngine
        prepared_splits: list[tuple[uuid.UUID, Decimal, str, Decimal | None]] = []

        if item.split_type == SplitTypeHint.PERCENTAGE and item.percentage_splits:
            pct_inputs: list[tuple[uuid.UUID, Decimal | None]] = []
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
                pct_inputs.append((p_uids[0], Decimal(str(ps.percentage))))

            try:
                prepared_splits = compute_splits(
                    amt_decimal, "PERCENTAGE", pct_inputs, payer_id=actual_payer_id
                )
            except ValueError:
                return _voice_rejection(
                    "UNRESOLVED_SPLIT",
                    "The split percentages didn't add up to 100%. Please speak again clearly.",
                    action_type="CREATE",
                )

        elif item.split_type == SplitTypeHint.EXACT and item.exact_splits:
            exact_inputs: list[tuple[uuid.UUID, Decimal | None]] = []
            for es in item.exact_splits:
                p_uids, unmatched_es, ambiguous_es = resolve_participants_strict(
                    [es.participant_raw], user_directory, user.id, group_mems
                )
                if ambiguous_es:
                    raw_name, tied_members = ambiguous_es[0]
                    opts = ", ".join(_format_candidate_label(m) for m in tied_members)
                    return _voice_rejection(
                        "AMBIGUOUS_FRIEND",
                        f"Found multiple friends matching '{raw_name}' ({opts}). Please speak again using their nickname.",
                        action_type="CREATE",
                        is_ambiguous=True,
                        status="AMBIGUOUS",
                    )
                if unmatched_es or not p_uids:
                    return _voice_rejection(
                        "UNKNOWN_FRIEND",
                        f"Couldn't find friend '{es.participant_raw}'. Please speak again using their nickname or add them first.",
                        action_type="CREATE",
                    )
                exact_inputs.append((p_uids[0], Decimal(str(es.amount))))

            try:
                prepared_splits = compute_splits(
                    amt_decimal, "EXACT", exact_inputs, payer_id=actual_payer_id
                )
            except ValueError:
                return _voice_rejection(
                    "UNRESOLVED_SPLIT",
                    f"The exact split amounts didn't add up to ₹{amt_decimal}. Please speak again clearly.",
                    action_type="CREATE",
                )

        elif item.split_type == SplitTypeHint.SHARES and item.share_splits:
            share_inputs: list[tuple[uuid.UUID, Decimal | None]] = []
            for ss in item.share_splits:
                p_uids, unmatched_ss, ambiguous_ss = resolve_participants_strict(
                    [ss.participant_raw], user_directory, user.id, group_mems
                )
                if ambiguous_ss:
                    raw_name, tied_members = ambiguous_ss[0]
                    opts = ", ".join(_format_candidate_label(m) for m in tied_members)
                    return _voice_rejection(
                        "AMBIGUOUS_FRIEND",
                        f"Found multiple friends matching '{raw_name}' ({opts}). Please speak again using their nickname.",
                        action_type="CREATE",
                        is_ambiguous=True,
                        status="AMBIGUOUS",
                    )
                if unmatched_ss or not p_uids:
                    return _voice_rejection(
                        "UNKNOWN_FRIEND",
                        f"Couldn't find friend '{ss.participant_raw}'. Please speak again using their nickname or add them first.",
                        action_type="CREATE",
                    )
                share_inputs.append((p_uids[0], Decimal(str(ss.shares))))

            try:
                prepared_splits = compute_splits(
                    amt_decimal, "SHARES", share_inputs, payer_id=actual_payer_id
                )
            except ValueError:
                return _voice_rejection(
                    "UNRESOLVED_SPLIT",
                    "Invalid share values provided. Please speak again clearly.",
                    action_type="CREATE",
                )

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

            # Fix Group Expense Defaulting: if group is specified and only caller ("me") or no one was named,
            # split equally across all members of the group!
            if group_mems and (not p_ids or p_ids == [user.id]):
                p_ids = [uuid.UUID(m["user_id"]) for m in group_mems]

            # If a friend paid (`actual_payer_id != user.id`), ensure caller (`user.id`) is included
            if actual_payer_id != user.id:
                if not p_ids or p_ids == [actual_payer_id]:
                    p_ids = [user.id, actual_payer_id]
                elif user.id not in p_ids and not group_mems:
                    p_ids.insert(0, user.id)

            # If caller paid (`actual_payer_id == user.id`) and command contains mutual split cues
            # ("split with", "with", "ke saath", "equally") rather than "on <friend>", ensure caller is included
            if actual_payer_id == user.id and user.id not in p_ids and p_ids:
                cmd_lower = (command_text or "").lower()
                is_on_friend = bool(re.search(r"\b(on|pe|ke\s+naam)\b", cmd_lower))
                has_mutual_cue = bool(
                    re.search(r"\b(split|with|saath|equally|half|50[-/\s]*50)\b", cmd_lower)
                )
                if has_mutual_cue and not is_on_friend:
                    p_ids.insert(0, user.id)

            if not p_ids or (not group_mems and actual_payer_id == user.id and p_ids == [user.id]):
                return _voice_rejection(
                    "LOW_CONFIDENCE",
                    "I wasn't sure who to split this expense with. Please speak again clearly.",
                    action_type="CREATE",
                )

            prepared_splits = compute_splits(
                amt_decimal,
                "EQUAL",
                [(uid, None) for uid in p_ids],
                payer_id=actual_payer_id,
            )

        prepared_shared.append(
            (item, resolved_gid, actual_payer_id, amt_decimal, item_date, prepared_splits)
        )

    # Commit all validated items atomically into unified `expenses` table
    created_self_objs: list[Expense] = []
    created_shared_ids: list[uuid.UUID] = []

    for item, amt_decimal, item_date in prepared_self:
        self_exp = Expense(
            payer_id=user.id,
            description=item.description.strip(),
            amount=amt_decimal,
            currency=item.currency,
            category=item.category or "general",
            expense_date=item_date,
            is_personal=True,
            is_deleted=False,
        )
        session.add(self_exp)
        created_self_objs.append(self_exp)

    for item, resolved_gid, actual_payer_id, amt_decimal, item_date, splits in prepared_shared:
        exp = Expense(
            group_id=resolved_gid,
            description=item.description.strip(),
            amount=amt_decimal,
            currency=item.currency,
            payer_id=actual_payer_id,
            category=item.category or "general",
            expense_date=item_date,
            is_personal=False,
            is_deleted=False,
        )
        session.add(exp)
        await session.flush()

        for uid, owed, stype, sval in splits:
            session.add(
                ExpenseSplit(
                    expense_id=exp.id,
                    user_id=uid,
                    owed_amount=owed,
                    split_type=stype,
                    split_value=sval,
                )
            )
        created_shared_ids.append(exp.id)

    await session.commit()

    created_self: list[SelfExpenseRead] = []
    for self_obj in created_self_objs:
        await session.refresh(self_obj)
        created_self.append(SelfExpenseRead.model_validate(self_obj))

    created_shared: list[ExpenseRead] = []
    if created_shared_ids:
        saved_exps = (
            await session.execute(
                select(Expense)
                .options(selectinload(Expense.splits))
                .where(Expense.id.in_(created_shared_ids))
            )
        ).scalars().all()
        exp_by_id = {e.id: e for e in saved_exps}
        name_map = {uuid.UUID(d["user_id"]): d["name"] for d in user_directory}
        for g in user_groups:
            for m in g["members"]:
                name_map.setdefault(uuid.UUID(m["user_id"]), m["name"])

        for eid in created_shared_ids:
            saved_e = exp_by_id.get(eid)
            if saved_e:
                model = ExpenseRead.model_validate(saved_e)
                model.payer_name = name_map.get(saved_e.payer_id, user.name)
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
    """Deterministic handler for bidirectional voice SETTLE commands ('I paid Sahaj' & 'Sahaj paid me')."""
    s_item = stage1_output.settle_item
    if not s_item:
        return _voice_rejection(
            "LOW_CONFIDENCE",
            "Couldn't catch settlement details. Please speak again clearly.",
            action_type="SETTLE",
        )

    settle_amt = Decimal(str(s_item.amount)).quantize(Decimal("0.01"))
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

    payer_raw_clean = (s_item.payer_raw or "me").strip()
    payee_raw_clean = (s_item.payee_raw or "").strip()

    # Case 1: Friend paid caller ("Sahaj paid me 200" -> payer_raw="Sahaj", payee_raw="me")
    if payer_raw_clean.lower() not in SELF_TOKENS and payee_raw_clean.lower() in SELF_TOKENS:
        friend_raw = payer_raw_clean
        caller_is_payer = False
    else:
        # Case 2: Caller paid friend ("I paid Sahaj 200" -> payer_raw="me", payee_raw="Sahaj")
        friend_raw = payee_raw_clean
        caller_is_payer = True

    friend_ids, unmatched_friends, ambiguous_friends = resolve_participants_strict(
        [friend_raw], user_directory, user.id
    )
    if ambiguous_friends:
        raw_name, tied_members = ambiguous_friends[0]
        opts = ", ".join(_format_candidate_label(m) for m in tied_members)
        return _voice_rejection(
            "AMBIGUOUS_FRIEND",
            f"Found multiple friends matching '{raw_name}' ({opts}). Please speak again using their nickname.",
            action_type="SETTLE",
            is_ambiguous=True,
            status="AMBIGUOUS",
        )
    target_friend_id = next((pid for pid in friend_ids if pid != user.id), None)
    if unmatched_friends or not target_friend_id:
        return _voice_rejection(
            "UNKNOWN_FRIEND",
            f"Couldn't find friend '{friend_raw}' to settle with. Please speak again using their nickname.",
            action_type="SETTLE",
        )

    actual_payer_id = user.id if caller_is_payer else target_friend_id
    actual_payee_id = target_friend_id if caller_is_payer else user.id

    settlement = Settlement(
        payer_id=actual_payer_id,
        payee_id=actual_payee_id,
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

    friend_user = (await session.execute(select(User).where(User.id == target_friend_id))).scalars().first()
    friend_name = friend_user.name if friend_user else friend_raw
    payer_name = user.name if caller_is_payer else friend_name
    payee_name = friend_name if caller_is_payer else user.name
    conf = (
        stage1_output.spoken_confirmation
        or f"Done! Settle up of ₹{settle_amt} from {payer_name} to {payee_name} recorded."
    )

    return VoiceIngestResponse(
        status="SUCCESS",
        action_type="SETTLE",
        count=1,
        settlement=SettlementRead(
            id=settlement.id,
            payer_id=settlement.payer_id,
            payer_name=payer_name,
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


async def _apply_update_or_delete(
    session: AsyncSession,
    user: User,
    action_type: str,
    target_id: uuid.UUID,
    new_description: str | None,
    new_amount: Decimal | None,
    new_category: str | None,
    spoken_confirmation: str | None,
    new_expense_date: datetime | None = None,
    force_description: bool = False,
) -> VoiceIngestResponse | None:
    """Apply UPDATE or DELETE on a verified Expense record owned/accessible by `user`, preserving split ratios."""
    target_exp = (
        await session.execute(
            select(Expense)
            .options(selectinload(Expense.splits))
            .where(Expense.id == target_id, Expense.is_deleted.is_(False))
        )
    ).scalars().first()
    if not target_exp:
        return None

    # Enforce strict ownership/access authorization
    if target_exp.is_personal:
        if target_exp.payer_id != user.id:
            return None
    else:
        has_access = (target_exp.payer_id == user.id) or any(
            s.user_id == user.id for s in target_exp.splits
        )
        if not has_access and target_exp.group_id is not None:
            grp_mem = (
                await session.execute(
                    select(GroupMember.id).where(
                        GroupMember.group_id == target_exp.group_id,
                        GroupMember.user_id == user.id,
                    )
                )
            ).scalars().first()
            has_access = bool(grp_mem)
        if not has_access:
            return None

    if action_type == "UPDATE":
        if new_description and new_description.strip():
            cand_desc = new_description.strip()
            cur_desc = target_exp.description.strip()
            if force_description or cand_desc.lower() not in cur_desc.lower():
                target_exp.description = cand_desc
        if new_amount is not None and new_amount > Decimal("0.00"):
            target_exp.amount = new_amount
            if not target_exp.is_personal and target_exp.splits:
                recalculate_splits_on_amount_change(
                    new_amount=target_exp.amount,
                    existing_splits=target_exp.splits,
                    payer_id=target_exp.payer_id,
                )
        if new_category and (new_category.strip().lower() != "general" or not target_exp.category):
            target_exp.category = new_category.strip()
        if new_expense_date is not None:
            target_exp.expense_date = new_expense_date

        target_exp.updated_at = datetime.now(UTC)
        await session.commit()

        target_exp = (
            await session.execute(
                select(Expense)
                .options(selectinload(Expense.splits))
                .where(Expense.id == target_id)
                .execution_options(populate_existing=True)
            )
        ).scalars().first()

        conf = spoken_confirmation or f"Done! Updated {target_exp.description} to ₹{target_exp.amount}."
        if target_exp.is_personal:
            return VoiceIngestResponse(
                status="SUCCESS",
                action_type="UPDATE",
                count=1,
                self_expenses=[SelfExpenseRead.model_validate(target_exp)],
                spoken_confirmation=conf,
                is_ambiguous=False,
            )
        model = ExpenseRead.model_validate(target_exp)
        if target_exp.payer_id == user.id:
            model.payer_name = user.name
        else:
            payer_u = (
                await session.execute(select(User.name).where(User.id == target_exp.payer_id))
            ).scalars().first()
            model.payer_name = payer_u or "Someone"
        return VoiceIngestResponse(
            status="SUCCESS",
            action_type="UPDATE",
            count=1,
            expenses=[model],
            spoken_confirmation=conf,
            is_ambiguous=False,
        )

    if action_type == "DELETE":
        target_exp.is_deleted = True
        target_exp.updated_at = datetime.now(UTC)
        await session.commit()
        return VoiceIngestResponse(
            status="SUCCESS",
            action_type="DELETE",
            count=1,
            spoken_confirmation=spoken_confirmation or f"Done! Deleted {target_exp.description}.",
            is_ambiguous=False,
        )

    return None


async def process_voice_pipeline(
    session: AsyncSession,
    user: User,
    command_text: str | None = None,
    client_timestamp_iso: str | None = None,
    group_id: uuid.UUID | None = None,
    audio_bytes: bytes | None = None,
    audio_mime_type: str = "audio/m4a",
) -> VoiceIngestResponse:
    """
    Single-Pass Multimodal Voice & Text Execution Engine:
    - Accepts either `command_text` or raw `audio_bytes` in one Gemini call with full directory/group context.
    - Only queries `get_user_balances` when `intent == GET`.
    - Handles CREATE, MULTI_CREATE, SETTLE, GET, UPDATE, and DELETE deterministically.
    """
    client = get_genai_client()

    user_directory = await get_user_directory(session, user)
    user_groups = await get_user_groups_context(session, user)

    # Fetch recent 20 modifiable expenses (including shared expenses paid by friends/groups)
    caller_split_sub = select(ExpenseSplit.id).where(
        ExpenseSplit.expense_id == Expense.id,
        ExpenseSplit.user_id == user.id,
    )
    caller_group_sub = select(GroupMember.id).where(
        GroupMember.group_id == Expense.group_id,
        GroupMember.user_id == user.id,
    )
    recent_exps = (
        await session.execute(
            select(Expense)
            .where(
                Expense.is_deleted.is_(False),
                or_(
                    Expense.payer_id == user.id,
                    and_(
                        Expense.is_personal.is_(False),
                        or_(caller_split_sub.exists(), caller_group_sub.exists()),
                    ),
                ),
            )
            .order_by(Expense.expense_date.desc(), Expense.created_at.desc())
            .limit(20)
        )
    ).scalars().all()

    now_ist = client_timestamp_iso or datetime.now(IST).isoformat()
    directory_str = json.dumps(
        [
            {
                "user_id": d["user_id"],
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
        [
            {
                "group_id": g["group_id"],
                "name": g["name"],
                "members": [m["name"] for m in g["members"]],
            }
            for g in user_groups
        ],
        ensure_ascii=False,
    )
    recent_str = json.dumps(
        [
            {
                "id": str(e.id),
                "description": e.description,
                "amount": float(e.amount),
                "date": e.expense_date.isoformat(),
                "category": e.category,
                "scope": "SELF" if e.is_personal else "SHARED",
            }
            for e in recent_exps
        ],
        ensure_ascii=False,
    )

    context_header = (
        f"Current User: {user.name} (ID: {user.id})\n"
        f"Client Timestamp (IST): {now_ist}\n"
        f"Active Group Filter ID: {group_id}\n"
        f"Known Friends & Directory: {directory_str}\n"
        f"Known Groups: {groups_str}\n"
        f"Recent Modifiable Expenses: {recent_str}\n"
    )

    if audio_bytes is not None:
        contents: Any = [
            types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime_type),
            context_header + "Extract the structured voice command from this audio recording.",
        ]
    else:
        contents = context_header + f"Voice Command: {command_text}"

    try:
        stage1_response = await client.aio.models.generate_content(
            model=settings.STAGE1_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=STAGE_1_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=StageOneOutput,
                temperature=0.1,
            ),
        )
        raw_text = (stage1_response.text or "").strip()
    except Exception:
        return _voice_rejection(
            "AI_SERVICE_ERROR",
            "Couldn't process voice command right now. Please speak again clearly.",
            action_type="UNKNOWN",
        )

    cleaned_marker = raw_text.strip("[](). ").upper()
    if not raw_text or "[UNCLEAR]" in raw_text.upper() or cleaned_marker in {
        "UNCLEAR",
        "INAUDIBLE",
        "NO_SPEECH",
        "SILENCE",
    }:
        return _voice_rejection(
            "UNCLEAR_AUDIO",
            "Sound was not clear. Please speak again clearly.",
            action_type="UNKNOWN",
        )

    try:
        stage1_output = StageOneOutput.model_validate_json(raw_text)
    except Exception:
        return _voice_rejection(
            "UNCLEAR_AUDIO",
            "Sound was not clear. Please speak again clearly.",
            action_type="UNKNOWN",
        )

    stage1_conf = (stage1_output.confidence or "HIGH").strip().upper()
    if stage1_output.intent == IntentType.UNKNOWN:
        msg = (
            stage1_output.spoken_confirmation
            or "Sound was not clear or I couldn't understand that. Please speak again clearly."
        )
        return _voice_rejection("LOW_CONFIDENCE", msg, action_type="UNKNOWN")

    if stage1_conf == "AMBIGUOUS" or len(stage1_output.ambiguous_candidate_ids) >= 2:
        msg = (
            stage1_output.clarification_question
            or stage1_output.spoken_confirmation
            or "Found multiple matching records. Please specify which one you mean."
        )
        return VoiceIngestResponse(
            status="AMBIGUOUS",
            action_type=str(stage1_output.intent),
            count=0,
            spoken_confirmation=stage1_output.spoken_confirmation or msg,
            is_ambiguous=True,
            clarification_prompt=msg,
            error_code="AMBIGUOUS_CANDIDATE",
        )

    if stage1_conf != "HIGH":
        msg = (
            stage1_output.spoken_confirmation
            or "I wasn't completely sure what you said. Please speak again clearly."
        )
        return _voice_rejection("LOW_CONFIDENCE", msg, action_type=str(stage1_output.intent))

    # 1. CREATE / MULTI_CREATE
    if stage1_output.intent in (IntentType.CREATE, IntentType.MULTI_CREATE):
        return await _handle_create_items(
            session=session,
            user=user,
            stage1_output=stage1_output,
            user_directory=user_directory,
            user_groups=user_groups,
            explicit_group_id=group_id,
            command_text=command_text,
        )

    # 2. SETTLE
    if stage1_output.intent == IntentType.SETTLE:
        return await _handle_settle(
            session=session,
            user=user,
            stage1_output=stage1_output,
            user_directory=user_directory,
            user_groups=user_groups,
            explicit_group_id=group_id,
        )

    # 3. GET (Lazy balance query only when intent is GET!)
    if stage1_output.intent == IntentType.GET:
        user_balances = await get_user_balances(session, user.id, group_id=group_id)
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

    # 4. UPDATE / DELETE (Single-Pass Direct Resolution with DB Fallback)
    action_str = str(stage1_output.intent)
    new_desc = stage1_output.items[0].description if stage1_output.items else None
    new_amt = (
        Decimal(str(stage1_output.items[0].amount)).quantize(Decimal("0.01"))
        if stage1_output.items and stage1_output.items[0].amount > 0
        else None
    )
    new_cat = stage1_output.items[0].category if stage1_output.items else None
    new_dt = (
        resolve_expense_date(stage1_output.items[0].expense_date_iso)
        if stage1_output.items and stage1_output.items[0].expense_date_iso
        else None
    )

    if stage1_output.target_record_id:
        try:
            direct_tid = uuid.UUID(stage1_output.target_record_id)
            direct_res = await _apply_update_or_delete(
                session=session,
                user=user,
                action_type=action_str,
                target_id=direct_tid,
                new_description=new_desc,
                new_amount=new_amt,
                new_category=new_cat,
                spoken_confirmation=stage1_output.spoken_confirmation,
                new_expense_date=new_dt,
            )
            if direct_res is not None:
                return direct_res
        except ValueError:
            pass

    # Fallback DB candidate search when target_record_id was not resolved in single pass
    search_keywords: list[str] = []
    target_dt: datetime | None = None
    window_days = 3
    amt_hint: float | None = None

    if stage1_output.search_query:
        search_keywords = stage1_output.search_query.description_keywords
        window_days = stage1_output.search_query.date_window_days or 3
        amt_hint = stage1_output.search_query.amount_hint
        if stage1_output.search_query.date_target_iso:
            target_dt = resolve_expense_date(stage1_output.search_query.date_target_iso)
    elif stage1_output.items:
        search_keywords = [stage1_output.items[0].description]

    self_cands, shared_cands = await search_db_candidates(
        session=session,
        user_id=user.id,
        keywords=search_keywords,
        target_date=target_dt,
        window_days=window_days,
        amount_hint=amt_hint,
    )
    all_cands = self_cands + shared_cands

    if not all_cands:
        return _voice_rejection(
            "LOW_CONFIDENCE",
            "Couldn't find a matching expense to modify. Please speak again clearly.",
            action_type=action_str,
        )

    # Deterministic single-candidate short-circuit when not explicitly marked complex
    if len(all_cands) == 1 and not stage1_output.is_complex:
        only_res = await _apply_update_or_delete(
            session=session,
            user=user,
            action_type=action_str,
            target_id=all_cands[0].id,
            new_description=new_desc,
            new_amount=new_amt,
            new_category=new_cat,
            spoken_confirmation=stage1_output.spoken_confirmation,
            new_expense_date=new_dt,
        )
        if only_res is not None:
            return only_res

    # Stage 2 Disambiguation when 2+ candidates exist or is_complex=True
    try:
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
                system_instruction=(
                    "Match the user's UPDATE or DELETE intent against the candidate expenses. "
                    "If exactly 1 candidate matches with HIGH confidence, return its target_record_id. "
                    "If 2+ candidates conflict, set confidence='AMBIGUOUS' and list ambiguous_candidate_ids."
                ),
                response_mime_type="application/json",
                response_schema=StageTwoResolution,
                temperature=0.1,
            ),
        )
        stage2_output = StageTwoResolution.model_validate_json(stage2_response.text or "")
    except Exception:
        if len(all_cands) == 1:
            only_res = await _apply_update_or_delete(
                session=session,
                user=user,
                action_type=action_str,
                target_id=all_cands[0].id,
                new_description=new_desc,
                new_amount=new_amt,
                new_category=new_cat,
                spoken_confirmation=stage1_output.spoken_confirmation,
                new_expense_date=new_dt,
            )
            if only_res:
                return only_res
        opts = ", ".join(f"{c.description} (₹{c.amount})" for c in all_cands[:3])
        msg = f"Found multiple matching expenses ({opts}). Which one would you like to {action_str.lower()}?"
        return VoiceIngestResponse(
            status="AMBIGUOUS",
            action_type=action_str,
            count=0,
            spoken_confirmation=msg,
            is_ambiguous=True,
            clarification_prompt=msg,
            error_code="AMBIGUOUS_CANDIDATE",
        )

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
            valid_ids = {c.id for c in all_cands}
            if target_id in valid_ids:
                s2_dt = (
                    resolve_expense_date(stage2_output.expense_date_iso)
                    if stage2_output.expense_date_iso
                    else new_dt
                )
                applied = await _apply_update_or_delete(
                    session=session,
                    user=user,
                    action_type=stage2_output.action_type,
                    target_id=target_id,
                    new_description=stage2_output.description or new_desc,
                    new_amount=stage2_output.amount if stage2_output.amount > 0 else new_amt,
                    new_category=stage2_output.category or new_cat,
                    spoken_confirmation=stage2_output.spoken_confirmation,
                    new_expense_date=s2_dt,
                )
                if applied is not None:
                    return applied
        except ValueError:
            pass

    return _voice_rejection(
        "LOW_CONFIDENCE",
        stage2_output.clarification_question
        or stage2_output.spoken_confirmation
        or "Couldn't find that expense to modify. Please speak again clearly.",
        action_type=stage2_output.action_type,
    )


# Alias for single-pass command execution
process_voice_command = process_voice_pipeline
