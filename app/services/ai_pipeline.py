"""AI Pipeline: Direct-to-Ledger Voice Execution Engine (gemini-3.8-flash -> gemini-4-hydrogen)."""

import json
import re
import uuid
import zoneinfo
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from google import genai
from google.genai import types
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models.expense import Expense, ExpenseSplit
from app.models.group import Group, GroupMember
from app.models.settlement import Settlement
from app.models.user import Friendship, User
from app.prompts.stage1_parser import STAGE_1_SYSTEM_PROMPT
from app.prompts.stage2_matcher import STAGE_2_SYSTEM_PROMPT
from app.schemas.ai import IntentType, SplitTypeHint, StageOneOutput, StageTwoResolution
from app.schemas.ledger import ExpenseRead
from app.schemas.settlement import SettlementRead
from app.schemas.voice import VoiceIngestResponse
from app.services.ledger_service import (
    calculate_equal_splits,
    calculate_percentage_splits,
    get_user_balances,
)

# IST Timezone (+05:30)
IST = zoneinfo.ZoneInfo("Asia/Kolkata")


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
    """Build directory of current user and friends with nicknames and phonetic aliases."""
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
    result = await session.execute(query)
    friendships = result.scalars().all()

    friend_ids = [f.friend_id for f in friendships]
    if friend_ids:
        friends_query = select(User).where(User.id.in_(friend_ids))
        f_users = (await session.execute(friends_query)).scalars().all()
        user_map = {u.id: u for u in f_users}

        for f in friendships:
            friend_user = user_map.get(f.friend_id)
            if friend_user:
                custom_aliases = [
                    a.strip() for a in (f.aliases or []) if a and a.strip()
                ]

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
    result = await session.execute(query)
    groups = result.scalars().all()

    groups_data = []
    for g in groups:
        # Load members
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


def resolve_group_id(group_raw: str | None, user_groups: list[dict[str, Any]]) -> uuid.UUID | None:
    """Fuzzy match raw spoken group name to an actual group ID."""
    if not group_raw:
        return None
    raw_clean = group_raw.strip().lower()
    for g in user_groups:
        g_name = g["name"].lower()
        if raw_clean == g_name or raw_clean in g_name or g_name in raw_clean:
            return uuid.UUID(g["group_id"])
        for alias in g.get("aliases", []):
            if raw_clean == alias or raw_clean in alias or alias in raw_clean:
                return uuid.UUID(g["group_id"])
    return None


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

    # Tier 1: Exact custom nickname or alias match
    if raw_clean in custom_labels:
        return (1, -len(raw_clean))

    # Tier 2: Exact User.name match
    if real_name and raw_clean == real_name:
        return (2, -len(raw_clean))

    # Tier 3: Whole-word match against custom nickname/aliases
    raw_word_pat = re.compile(rf"\b{re.escape(raw_clean)}\b")
    for lbl in custom_labels:
        if raw_word_pat.search(lbl) or re.search(rf"\b{re.escape(lbl)}\b", raw_clean):
            return (3, -len(lbl))

    # Tier 4: Whole-word match against User.name
    if real_name and (
        raw_word_pat.search(real_name) or re.search(rf"\b{re.escape(real_name)}\b", raw_clean)
    ):
        return (4, -len(real_name))

    # Tier 5: Substring fallback (length >= 3 to avoid short-token collisions like 'ram' in 'vikram')
    if len(raw_clean) >= 3:
        for lbl in custom_labels:
            if len(lbl) >= 3 and (raw_clean in lbl or lbl in raw_clean):
                return (5, -len(lbl))
        if real_name and len(real_name) >= 3 and (raw_clean in real_name or real_name in raw_clean):
            return (5, -len(real_name))

    return None


def resolve_participant_ids(
    raw_names: list[str],
    user_directory: list[dict[str, Any]],
    caller_id: uuid.UUID,
    group_members: list[dict[str, Any]] | None = None,
) -> list[uuid.UUID]:
    """Map spoken raw participant names to verified User IDs using ranked alias & name matching."""
    # Combine directory with group members if available
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

    for raw in raw_names:
        raw_clean = raw.strip().lower()
        if not raw_clean:
            continue
        if raw_clean in ["me", "myself", "i", "maine", "mera"]:
            matched_ids.add(caller_id)
            continue

        best_score: tuple[int, int, int] | None = None
        best_member_id: uuid.UUID | None = None

        for idx, member in enumerate(combined_directory):
            if member.get("is_caller"):
                continue
            score = _score_member_match(raw_clean, member)
            if score is not None:
                ranked = (score[0], score[1], idx)
                if best_score is None or ranked < best_score:
                    best_score = ranked
                    best_member_id = uuid.UUID(member["user_id"])

        if best_member_id is not None:
            matched_ids.add(best_member_id)

    # Fallback to caller only if no participant matched, or caller was explicitly included
    if not matched_ids:
        matched_ids.add(caller_id)
    elif caller_id not in matched_ids and any(
        r.strip().lower() in ["me", "myself", "i", "all", "sab"] for r in raw_names
    ):
        matched_ids.add(caller_id)

    return list(matched_ids)


async def search_verified_candidates(
    session: AsyncSession,
    user_id: uuid.UUID,
    keywords: list[str],
    target_date: datetime | None = None,
    window_days: int = 3,
) -> list[Expense]:
    """Search verified expenses for matching keywords and dates."""
    verified_query = select(Expense).where(Expense.is_deleted.is_(False))
    if target_date:
        if target_date.tzinfo is None:
            target_date = target_date.replace(tzinfo=UTC)
        start_date = target_date - timedelta(days=window_days)
        end_date = target_date + timedelta(days=window_days)
        verified_query = verified_query.where(
            Expense.expense_date >= start_date,
            Expense.expense_date <= end_date,
        )

    if keywords:
        filters = []
        for kw in keywords:
            filters.append(Expense.description.ilike(f"%{kw}%"))
            filters.append(Expense.category.ilike(f"%{kw}%"))
        verified_query = verified_query.where(or_(*filters))

    verified_res = await session.execute(verified_query)
    return list(verified_res.scalars().all())


async def search_db_candidates(
    session: AsyncSession,
    user_id: uuid.UUID,
    keywords: list[str],
    target_date: datetime | None = None,
    window_days: int = 3,
) -> tuple[list[Any], list[Expense]]:
    """Search candidates across staged (empty) and verified expenses for backward compatibility."""
    verified = await search_verified_candidates(session, user_id, keywords, target_date, window_days)
    return [], verified


async def process_voice_pipeline(
    session: AsyncSession,
    user: User,
    command_text: str,
    client_timestamp_iso: str | None = None,
    group_id: uuid.UUID | None = None,
) -> VoiceIngestResponse:
    """
    Direct-to-Ledger Voice Processing Pipeline:
    1. Context builder: Injects friends, registered aliases, user groups & member rosters.
    2. Stage 1 (gemini-3.8-flash): Fast intent routing.
    3. Direct Ledger Action:
       - CREATE / MULTI_CREATE: Directly creates Expense + ExpenseSplit rows (Equal or Percentage Hare-Niemeyer).
       - SETTLE: Directly creates Settlement row and clears net balance.
       - UPDATE / DELETE: Disambiguates candidate, updates/deletes expense.
       - GET: Answers balance/spending queries conversationally.
    """
    client = get_genai_client()

    # Determine IST anchor timestamp
    if client_timestamp_iso:
        try:
            client_dt = datetime.fromisoformat(client_timestamp_iso)
        except ValueError:
            client_dt = datetime.now(IST)
    else:
        client_dt = datetime.now(IST)

    user_directory = await get_user_directory(session, user)
    user_groups = await get_user_groups_context(session, user)
    user_balances = await get_user_balances(session, user.id)

    # -------------------------------------------------------------
    # STAGE 1: Fast Audio / Text Perception & Intent Router
    # -------------------------------------------------------------
    stage1_prompt = f"""
Current Client Timestamp (IST): {client_dt.isoformat()}
Known Friends & Directory: {json.dumps([{'name': u['name'], 'nickname': u.get('nickname'), 'aliases': u.get('aliases')} for u in user_directory])}
User Groups & Members: {json.dumps([{'name': g['name'], 'members': [m['name'] for m in g['members']]} for g in user_groups])}

User Voice Utterance:
"{command_text}"
"""

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

    # -------------------------------------------------------------
    # FAST PATH: Simple CREATE or MULTI_CREATE with is_complex == False
    # (Directly committed to the verified ledger with zero staging ceremony!)
    # -------------------------------------------------------------
    if not stage1_output.is_complex and stage1_output.intent in [
        IntentType.CREATE,
        IntentType.MULTI_CREATE,
    ]:
        created_expenses: list[ExpenseRead] = []

        for item in stage1_output.items:
            # Resolve group if any
            resolved_gid = group_id or resolve_group_id(item.group_raw, user_groups)

            # Resolve participants
            group_mems = None
            if resolved_gid:
                matched_g = next((g for g in user_groups if g["group_id"] == str(resolved_gid)), None)
                if matched_g:
                    group_mems = matched_g["members"]

            p_ids = resolve_participant_ids(item.participants_raw, user_directory, user.id, group_mems)
            if not p_ids and group_mems:
                p_ids = [uuid.UUID(m["user_id"]) for m in group_mems]
            if not p_ids:
                p_ids = [user.id]

            # Expense date
            item_date = datetime.now(UTC)
            if item.expense_date_iso:
                try:
                    parsed_d = datetime.fromisoformat(item.expense_date_iso)
                    item_date = parsed_d.replace(tzinfo=UTC)
                except ValueError:
                    pass

            amt_decimal = Decimal(str(item.amount))

            expense = Expense(
                group_id=resolved_gid,
                description=item.description,
                amount=amt_decimal,
                currency=item.currency,
                payer_id=user.id,
                category=item.category or "general",
                expense_date=item_date,
                is_deleted=False,
            )
            session.add(expense)
            await session.flush()

            # Equal split calculation
            splits_calc = calculate_equal_splits(amt_decimal, p_ids, payer_id=user.id)
            for uid, owed in splits_calc:
                split = ExpenseSplit(
                    expense_id=expense.id,
                    user_id=uid,
                    owed_amount=owed,
                    split_type="EQUAL",
                )
                session.add(split)

            await session.commit()

            # Load created with splits
            fetch_q = (
                select(Expense)
                .options(selectinload(Expense.splits))
                .where(Expense.id == expense.id)
            )
            saved = (await session.execute(fetch_q)).scalars().first()
            created_expenses.append(ExpenseRead.model_validate(saved))

        return VoiceIngestResponse(
            status="SUCCESS",
            action_type="CREATE",
            count=len(created_expenses),
            expenses=created_expenses,
            spoken_confirmation=stage1_output.spoken_confirmation
            or f"Done! Added {len(created_expenses)} expense(s) to your ledger.",
            is_ambiguous=False,
        )

    # -------------------------------------------------------------
    # FAST PATH: Voice SETTLE UP ("I paid Sahaj 250 via UPI")
    # -------------------------------------------------------------
    if stage1_output.intent == IntentType.SETTLE and stage1_output.settle_item:
        s_item = stage1_output.settle_item
        payee_ids = resolve_participant_ids([s_item.payee_raw], user_directory, user.id)
        # Exclude caller from payee
        target_payee_id = next((pid for pid in payee_ids if pid != user.id), None)
        if not target_payee_id and user_balances.balances:
            # If payee name was ambiguous, use the friend with largest debt
            target_payee_id = user_balances.balances[0].user_id

        if target_payee_id:
            settle_amt = Decimal(str(s_item.amount))
            resolved_gid = group_id or resolve_group_id(s_item.group_raw, user_groups)

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

    # -------------------------------------------------------------
    # FAST PATH: Voice GET BALANCE ("What do I owe Sahaj?")
    # -------------------------------------------------------------
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
    # STAGE 2 PATH: Complex reasoning / Percentage Splits / UPDATE / DELETE
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

    verified_cands = await search_verified_candidates(
        session=session,
        user_id=user.id,
        keywords=search_keywords,
        target_date=target_dt,
    )

    stage2_context = {
        "stage1_extraction": stage1_output.model_dump(),
        "candidate_verified_expenses": [
            {
                "id": str(e.id),
                "description": e.description,
                "amount": float(e.amount),
                "date": e.expense_date.isoformat(),
                "category": e.category,
                "payer_id": str(e.payer_id),
            }
            for e in verified_cands
        ],
        "user_directory": user_directory,
        "user_groups": user_groups,
        "user_balances": [
            {"user_id": str(b.user_id), "user_name": b.user_name, "net_balance": float(b.net_balance)}
            for b in user_balances.balances
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

    # Check for Ambiguity
    if stage2_output.confidence == "AMBIGUOUS" or len(stage2_output.ambiguous_candidate_ids) >= 2:
        return VoiceIngestResponse(
            status="AMBIGUOUS",
            action_type=stage2_output.action_type,
            count=0,
            spoken_confirmation=stage2_output.spoken_confirmation,
            is_ambiguous=True,
            clarification_prompt=stage2_output.clarification_question
            or "Found multiple matching records. Please specify which one you want to update.",
        )

    # -------------------------------------------------------------
    # EXECUTE STAGE 2 RESOLUTION DIRECTLY ON THE LEDGER
    # -------------------------------------------------------------

    # Case A: Complex CREATE (with Percentage splits or Group resolution)
    if stage2_output.action_type == "CREATE":
        resolved_gid = None
        if stage2_output.group_id:
            try:
                resolved_gid = uuid.UUID(stage2_output.group_id)
            except ValueError:
                pass
        if not resolved_gid and stage1_output.items:
            resolved_gid = group_id or resolve_group_id(stage1_output.items[0].group_raw, user_groups)

        exp_amt = stage2_output.amount
        exp_desc = stage2_output.description or (stage1_output.items[0].description if stage1_output.items else "Expense")

        payer_id = user.id
        if stage2_output.payer_user_id:
            try:
                payer_id = uuid.UUID(stage2_output.payer_user_id)
            except ValueError:
                pass

        expense = Expense(
            group_id=resolved_gid,
            description=exp_desc,
            amount=exp_amt,
            currency=stage2_output.currency or "INR",
            payer_id=payer_id,
            category=stage2_output.category or "general",
            expense_date=datetime.now(UTC),
            is_deleted=False,
        )
        session.add(expense)
        await session.flush()

        # Handle percentage splits from Stage 1 or Stage 2
        first_item = stage1_output.items[0] if stage1_output.items else None
        if first_item and first_item.split_type == SplitTypeHint.PERCENTAGE and first_item.percentage_splits:
            # Map percentages to participant IDs
            pct_tuples: list[tuple[uuid.UUID, Decimal]] = []
            for ps in first_item.percentage_splits:
                p_uid_list = resolve_participant_ids([ps.participant_raw], user_directory, user.id)
                if p_uid_list:
                    pct_tuples.append((p_uid_list[0], Decimal(str(ps.percentage))))

            if pct_tuples and sum(pct for _, pct in pct_tuples) == Decimal("100.00"):
                calc_pct_splits = calculate_percentage_splits(exp_amt, pct_tuples, payer_id=payer_id)
                for uid, owed in calc_pct_splits:
                    session.add(
                        ExpenseSplit(
                            expense_id=expense.id,
                            user_id=uid,
                            owed_amount=owed,
                            split_type="PERCENTAGE",
                        )
                    )
            else:
                # Fallback to resolved splits from Stage 2
                for s in stage2_output.resolved_splits:
                    if s.user_id:
                        session.add(
                            ExpenseSplit(
                                expense_id=expense.id,
                                user_id=uuid.UUID(s.user_id),
                                owed_amount=s.owed_amount,
                                split_type="EXACT",
                            )
                        )
        elif stage2_output.resolved_splits and any(s.user_id for s in stage2_output.resolved_splits):
            for s in stage2_output.resolved_splits:
                if s.user_id:
                    session.add(
                        ExpenseSplit(
                            expense_id=expense.id,
                            user_id=uuid.UUID(s.user_id),
                            owed_amount=s.owed_amount,
                            split_type="EXACT",
                        )
                    )
        else:
            # Default equal split among resolved participants
            p_ids = [user.id]
            splits_calc = calculate_equal_splits(exp_amt, p_ids, payer_id=payer_id)
            for uid, owed in splits_calc:
                session.add(
                    ExpenseSplit(
                        expense_id=expense.id,
                        user_id=uid,
                        owed_amount=owed,
                        split_type="EQUAL",
                    )
                )

        await session.commit()

        fetch_q = (
            select(Expense)
            .options(selectinload(Expense.splits))
            .where(Expense.id == expense.id)
        )
        saved = (await session.execute(fetch_q)).scalars().first()

        return VoiceIngestResponse(
            status="SUCCESS",
            action_type="CREATE",
            count=1,
            expenses=[ExpenseRead.model_validate(saved)],
            spoken_confirmation=stage2_output.spoken_confirmation
            or f"Done! Recorded ₹{exp_amt} for {exp_desc}.",
            is_ambiguous=False,
        )

    # Case B: UPDATE existing expense
    if stage2_output.action_type == "UPDATE" and stage2_output.target_record_id:
        target_id = uuid.UUID(stage2_output.target_record_id)
        target_exp = (
            await session.execute(
                select(Expense)
                .options(selectinload(Expense.splits))
                .where(Expense.id == target_id)
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

            # Rebalance splits if amount changed
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

            return VoiceIngestResponse(
                status="SUCCESS",
                action_type="UPDATE",
                count=1,
                expenses=[ExpenseRead.model_validate(target_exp)],
                spoken_confirmation=stage2_output.spoken_confirmation
                or f"Done! Updated {target_exp.description} to ₹{target_exp.amount}.",
                is_ambiguous=False,
            )

    # Case C: DELETE existing expense
    if stage2_output.action_type == "DELETE" and stage2_output.target_record_id:
        target_id = uuid.UUID(stage2_output.target_record_id)
        target_exp = (
            await session.execute(select(Expense).where(Expense.id == target_id))
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

    # Case D: SETTLE via Stage 2
    if stage2_output.action_type == "SETTLE":
        target_payee_id = None
        if stage2_output.payee_user_id:
            try:
                target_payee_id = uuid.UUID(stage2_output.payee_user_id)
            except ValueError:
                pass
        if not target_payee_id and user_balances.balances:
            target_payee_id = user_balances.balances[0].user_id

        if target_payee_id:
            settlement = Settlement(
                payer_id=user.id,
                payee_id=target_payee_id,
                amount=stage2_output.amount,
                currency=stage2_output.currency or "INR",
                group_id=uuid.UUID(stage2_output.group_id) if stage2_output.group_id else None,
                payment_method="UPI",
                notes="Voice settle up",
                settlement_date=datetime.now(UTC),
                is_deleted=False,
            )
            session.add(settlement)
            await session.commit()
            await session.refresh(settlement)

            payee_user = (await session.execute(select(User).where(User.id == target_payee_id))).scalars().first()
            payee_name = payee_user.name if payee_user else "Friend"

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
                spoken_confirmation=stage2_output.spoken_confirmation
                or f"Done! Settled ₹{settlement.amount} with {payee_name}.",
                is_ambiguous=False,
            )

    # Fallback
    return VoiceIngestResponse(
        status="SUCCESS",
        action_type=stage2_output.action_type,
        spoken_confirmation=stage2_output.spoken_confirmation or "Command processed.",
        is_ambiguous=False,
    )
