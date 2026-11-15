"""Group, Friendship, Shared Activity Feeds, and Net Balance endpoints."""

import uuid
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.database import get_session
from app.models.expense import Expense
from app.models.group import Group, GroupMember
from app.models.settlement import Settlement
from app.models.user import Friendship, User
from app.schemas.ledger import (
    ActivityTimelineItem,
    BalanceSummary,
    BilateralActivityFeed,
    FriendCreate,
    FriendRead,
    GroupActivityFeed,
    GroupCreate,
    GroupDetailRead,
    GroupMemberAdd,
    GroupMemberRead,
    GroupRead,
)
from app.services.ledger_service import get_user_balances

router = APIRouter(tags=["Groups, Friends & Balances"])


# -------------------------------------------------------------
# GROUPS
# -------------------------------------------------------------


@router.post(
    "/groups",
    response_model=GroupRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new expense group",
)
async def create_group(
    payload: GroupCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> GroupRead:
    """Create group and automatically assign current user as admin member."""
    group = Group(
        name=payload.name,
        created_by=current_user.id,
    )
    session.add(group)
    await session.flush()

    member = GroupMember(
        group_id=group.id,
        user_id=current_user.id,
        role="admin",
    )
    session.add(member)
    await session.commit()
    await session.refresh(group)

    return GroupRead.model_validate(group)


@router.get(
    "/groups",
    response_model=list[GroupRead],
    summary="List groups current user is a member of",
)
async def list_user_groups(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[GroupRead]:
    """Retrieve all groups for active user."""
    query = (
        select(Group)
        .join(GroupMember, Group.id == GroupMember.group_id)
        .where(GroupMember.user_id == current_user.id)
    )
    result = await session.execute(query)
    groups = result.scalars().all()
    return [GroupRead.model_validate(g) for g in groups]


@router.get(
    "/groups/{group_id}",
    response_model=GroupDetailRead,
    summary="Get group profile with member list and total spending",
)
async def get_group_details(
    group_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> GroupDetailRead:
    """Fetch group metadata, member roster, and aggregate spending."""
    group = (await session.execute(select(Group).where(Group.id == group_id))).scalars().first()
    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found.")

    # Check caller membership
    mem_query = select(GroupMember).where(
        GroupMember.group_id == group_id, GroupMember.user_id == current_user.id
    )
    if not (await session.execute(mem_query)).scalars().first():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a member of this group.")

    # Fetch members
    members_res = await session.execute(
        select(GroupMember, User)
        .join(User, GroupMember.user_id == User.id)
        .where(GroupMember.group_id == group_id)
    )
    member_records = members_res.all()
    members_list = [
        GroupMemberRead(
            user_id=u.id,
            name=u.name,
            email=u.email,
            phone=u.phone,
            role=gm.role,
            joined_at=gm.joined_at,
        )
        for gm, u in member_records
    ]

    # Calculate total spending in group
    exp_res = await session.execute(
        select(Expense).where(Expense.group_id == group_id, Expense.is_deleted.is_(False))
    )
    expenses = exp_res.scalars().all()
    total_spending = sum((e.amount for e in expenses), Decimal("0.00"))

    return GroupDetailRead(
        id=group.id,
        name=group.name,
        invite_code=group.invite_code,
        created_by=group.created_by,
        created_at=group.created_at,
        members=members_list,
        total_spending=total_spending,
    )


@router.post(
    "/groups/{group_id}/members",
    response_model=GroupMemberRead,
    status_code=status.HTTP_201_CREATED,
    summary="Directly add a member to a group (by phone, email, or user ID)",
)
async def add_group_member(
    group_id: uuid.UUID,
    payload: GroupMemberAdd,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> GroupMemberRead:
    """Add a friend or contact directly into a group without requiring an invite code."""
    group = (await session.execute(select(Group).where(Group.id == group_id))).scalars().first()
    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found.")

    # Verify caller is a group member
    caller_mem = (
        await session.execute(
            select(GroupMember).where(
                GroupMember.group_id == group_id, GroupMember.user_id == current_user.id
            )
        )
    ).scalars().first()
    if not caller_mem:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only group members can add others.")

    ident = payload.identifier.strip()
    target_user: User | None = None

    # Check if UUID
    try:
        user_uuid = uuid.UUID(ident)
        target_user = (await session.execute(select(User).where(User.id == user_uuid))).scalars().first()
    except ValueError:
        pass

    # Check if email
    if not target_user and "@" in ident:
        target_user = (
            await session.execute(select(User).where(User.email == ident.lower()))
        ).scalars().first()
        if not target_user:
            uname = payload.name or ident.split("@")[0].title()
            target_user = User(email=ident.lower(), name=uname)
            session.add(target_user)
            await session.flush()

    # Check if phone
    if not target_user:
        target_user = (
            await session.execute(select(User).where(User.phone == ident))
        ).scalars().first()
        if not target_user:
            uname = payload.name or f"Friend {ident[-4:]}"
            fake_email = f"user_{ident.replace('+', '')}@bolosplit.app"
            target_user = User(email=fake_email, phone=ident, name=uname)
            session.add(target_user)
            await session.flush()

    # Check if already a member
    existing_mem = (
        await session.execute(
            select(GroupMember).where(
                GroupMember.group_id == group_id, GroupMember.user_id == target_user.id
            )
        )
    ).scalars().first()

    if not existing_mem:
        mem = GroupMember(group_id=group_id, user_id=target_user.id, role=payload.role)
        session.add(mem)

        # Also register reciprocal friendship between caller and target_user
        if target_user.id != current_user.id:
            for u1, u2, nick in [(current_user, target_user, target_user.name), (target_user, current_user, current_user.name)]:
                ship = (
                    await session.execute(
                        select(Friendship).where(
                            Friendship.user_id == u1.id, Friendship.friend_id == u2.id
                        )
                    )
                ).scalars().first()
                if not ship:
                    session.add(
                        Friendship(
                            user_id=u1.id,
                            friend_id=u2.id,
                            nickname=nick,
                            aliases=[u2.name.lower()],
                        )
                    )

        await session.commit()
    else:
        mem = existing_mem

    return GroupMemberRead(
        user_id=target_user.id,
        name=target_user.name,
        email=target_user.email,
        phone=target_user.phone,
        role=mem.role,
        joined_at=mem.joined_at,
    )


@router.post(
    "/groups/join/{invite_code}",
    response_model=GroupRead,
    summary="Join an existing group with invite code",
)
async def join_group(
    invite_code: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> GroupRead:
    """Join a shared group using a 6-character mobile invite code."""
    clean_code = invite_code.strip().upper()
    query = select(Group).where(Group.invite_code == clean_code)
    result = await session.execute(query)
    group = result.scalars().first()
    if not group:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid invite code.",
        )

    mem_query = select(GroupMember).where(
        GroupMember.group_id == group.id,
        GroupMember.user_id == current_user.id,
    )
    existing_mem = (await session.execute(mem_query)).scalars().first()
    if not existing_mem:
        member = GroupMember(
            group_id=group.id,
            user_id=current_user.id,
            role="member",
        )
        session.add(member)
        await session.commit()

    return GroupRead.model_validate(group)


@router.get(
    "/groups/{group_id}/expenses",
    response_model=GroupActivityFeed,
    summary="Get group activity feed (timeline of expenses & settlements)",
)
async def get_group_activity_feed(
    group_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> GroupActivityFeed:
    """Retrieve chronological timeline of expenses and settlements in a group."""
    group = (await session.execute(select(Group).where(Group.id == group_id))).scalars().first()
    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found.")

    # Fetch expenses with splits
    exp_query = (
        select(Expense)
        .options(selectinload(Expense.splits))
        .where(Expense.group_id == group_id, Expense.is_deleted.is_(False))
        .order_by(Expense.expense_date.desc())
        .offset(offset)
        .limit(limit)
    )
    expenses = (await session.execute(exp_query)).scalars().all()

    # Fetch settlements
    set_query = (
        select(Settlement)
        .where(Settlement.group_id == group_id, Settlement.is_deleted.is_(False))
        .order_by(Settlement.settlement_date.desc())
        .offset(offset)
        .limit(limit)
    )
    settlements = (await session.execute(set_query)).scalars().all()

    # Load user names
    user_ids = {e.payer_id for e in expenses} | {s.payer_id for s in settlements} | {s.payee_id for s in settlements}
    for e in expenses:
        user_ids.update(sp.user_id for sp in e.splits)
    names: dict[uuid.UUID, str] = {}
    if user_ids:
        u_res = await session.execute(select(User).where(User.id.in_(user_ids)))
        for u in u_res.scalars().all():
            names[u.id] = u.name

    timeline: list[ActivityTimelineItem] = []

    for e in expenses:
        # Determine your share and net impact
        your_split = next((sp for sp in e.splits if sp.user_id == current_user.id), None)
        your_share = your_split.owed_amount if your_split else Decimal("0.00")
        if e.payer_id == current_user.id:
            # You paid total e.amount, others owe (e.amount - your_share)
            impact = e.amount - your_share
        else:
            # Someone else paid, you owe your_share
            impact = -your_share

        timeline.append(
            ActivityTimelineItem(
                type="EXPENSE",
                id=e.id,
                date=e.expense_date,
                description=e.description,
                total_amount=e.amount,
                currency=e.currency,
                payer_id=e.payer_id,
                payer_name=names.get(e.payer_id, "Unknown"),
                group_id=e.group_id,
                your_share=your_share,
                your_net_impact=impact,
            )
        )

    for s in settlements:
        if s.payer_id == current_user.id:
            impact = s.amount  # You paid: debt reduced
        elif s.payee_id == current_user.id:
            impact = -s.amount  # You received money
        else:
            impact = Decimal("0.00")  # Between two other members

        timeline.append(
            ActivityTimelineItem(
                type="SETTLEMENT",
                id=s.id,
                date=s.settlement_date,
                description=f"{names.get(s.payer_id, 'Someone')} paid {names.get(s.payee_id, 'someone')} {s.amount}",
                total_amount=s.amount,
                currency=s.currency,
                payer_id=s.payer_id,
                payer_name=names.get(s.payer_id, "Unknown"),
                payee_id=s.payee_id,
                payee_name=names.get(s.payee_id, "Unknown"),
                group_id=s.group_id,
                your_net_impact=impact,
            )
        )

    # Sort combined timeline by date desc
    timeline.sort(key=lambda x: x.date, reverse=True)
    total_spending = sum((e.amount for e in expenses), Decimal("0.00"))

    return GroupActivityFeed(
        group_id=group.id,
        group_name=group.name,
        total_spending=total_spending,
        timeline=timeline,
    )


# -------------------------------------------------------------
# FRIENDS
# -------------------------------------------------------------


@router.post(
    "/friends",
    response_model=FriendRead,
    status_code=status.HTTP_201_CREATED,
    summary="Register a friend (by email or phone) with reciprocal visibility",
)
async def add_friend(
    payload: FriendCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> FriendRead:
    """Register a friend connection with phone/email and reciprocal visibility."""
    target_user: User | None = None

    if payload.friend_email:
        email = payload.friend_email.strip().lower()
        target_user = (await session.execute(select(User).where(User.email == email))).scalars().first()
        if not target_user:
            fname = payload.friend_name or email.split("@")[0].title()
            target_user = User(email=email, phone=payload.friend_phone, name=fname)
            session.add(target_user)
            await session.flush()
    elif payload.friend_phone:
        phone = payload.friend_phone.strip()
        target_user = (await session.execute(select(User).where(User.phone == phone))).scalars().first()
        if not target_user:
            fname = payload.friend_name or f"Friend {phone[-4:]}"
            fake_email = f"user_{phone.replace('+', '')}@bolosplit.app"
            target_user = User(email=fake_email, phone=phone, name=fname)
            session.add(target_user)
            await session.flush()
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either friend_email or friend_phone is required.",
        )

    clean_aliases = list(set([a.strip().lower() for a in payload.aliases if a.strip()]))
    if payload.nickname and payload.nickname.strip().lower() not in clean_aliases:
        clean_aliases.append(payload.nickname.strip().lower())
    if target_user.name.strip().lower() not in clean_aliases:
        clean_aliases.append(target_user.name.strip().lower())

    # 1. Forward friendship: caller -> friend
    ship1 = (
        await session.execute(
            select(Friendship).where(
                Friendship.user_id == current_user.id, Friendship.friend_id == target_user.id
            )
        )
    ).scalars().first()

    if ship1:
        ship1.nickname = payload.nickname
        ship1.aliases = clean_aliases
    else:
        ship1 = Friendship(
            user_id=current_user.id,
            friend_id=target_user.id,
            nickname=payload.nickname,
            aliases=clean_aliases,
        )
        session.add(ship1)

    # 2. Reciprocal friendship: friend -> caller (so friend sees caller in their friends list!)
    ship2 = (
        await session.execute(
            select(Friendship).where(
                Friendship.user_id == target_user.id, Friendship.friend_id == current_user.id
            )
        )
    ).scalars().first()

    if not ship2:
        caller_aliases = [current_user.name.lower()]
        ship2 = Friendship(
            user_id=target_user.id,
            friend_id=current_user.id,
            nickname=current_user.name,
            aliases=caller_aliases,
        )
        session.add(ship2)

    await session.commit()
    await session.refresh(ship1)

    return FriendRead(
        id=ship1.id,
        user_id=ship1.user_id,
        friend_id=ship1.friend_id,
        friend_name=target_user.name,
        friend_email=target_user.email,
        friend_phone=target_user.phone,
        nickname=ship1.nickname,
        aliases=ship1.aliases,
        created_at=ship1.created_at,
    )


@router.get(
    "/friends",
    response_model=list[FriendRead],
    summary="List friends and their registered voice aliases",
)
async def list_friends(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[FriendRead]:
    """Retrieve all friends with spoken alias mappings."""
    query = select(Friendship).where(Friendship.user_id == current_user.id)
    ships = (await session.execute(query)).scalars().all()

    friends_out: list[FriendRead] = []
    for s in ships:
        f_user = (
            await session.execute(select(User).where(User.id == s.friend_id))
        ).scalars().first()
        if f_user:
            friends_out.append(
                FriendRead(
                    id=s.id,
                    user_id=s.user_id,
                    friend_id=s.friend_id,
                    friend_name=f_user.name,
                    friend_email=f_user.email,
                    friend_phone=f_user.phone,
                    nickname=s.nickname,
                    aliases=s.aliases,
                    created_at=s.created_at,
                )
            )
    return friends_out


@router.get(
    "/friends/{friend_id}/expenses",
    response_model=BilateralActivityFeed,
    summary="Get 1-on-1 activity feed (all shared expenses and settlements with friend)",
)
async def get_bilateral_activity_feed(
    friend_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> BilateralActivityFeed:
    """Retrieve chronological 1-on-1 activity timeline and net balance with a friend."""
    friend = (await session.execute(select(User).where(User.id == friend_id))).scalars().first()
    if not friend:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Friend not found.")

    # 1. Fetch expenses where both are involved
    exp_query = (
        select(Expense)
        .options(selectinload(Expense.splits))
        .where(
            Expense.is_deleted.is_(False),
            or_(
                (Expense.payer_id == current_user.id),
                (Expense.payer_id == friend_id),
            ),
        )
        .order_by(Expense.expense_date.desc())
    )
    all_candidate_exps = (await session.execute(exp_query)).scalars().all()

    # Filter to expenses where the other person has a split
    shared_expenses: list[Expense] = []
    for exp in all_candidate_exps:
        has_current = any(sp.user_id == current_user.id for sp in exp.splits) or exp.payer_id == current_user.id
        has_friend = any(sp.user_id == friend_id for sp in exp.splits) or exp.payer_id == friend_id
        if has_current and has_friend:
            shared_expenses.append(exp)

    # 2. Fetch settlements between caller and friend
    set_query = (
        select(Settlement)
        .where(
            Settlement.is_deleted.is_(False),
            or_(
                (Settlement.payer_id == current_user.id) & (Settlement.payee_id == friend_id),
                (Settlement.payer_id == friend_id) & (Settlement.payee_id == current_user.id),
            ),
        )
        .order_by(Settlement.settlement_date.desc())
    )
    settlements = (await session.execute(set_query)).scalars().all()

    # Calculate net balance specifically with this friend
    net_balance = Decimal("0.00")
    timeline: list[ActivityTimelineItem] = []

    for exp in shared_expenses:
        if exp.payer_id == current_user.id:
            # Caller paid: friend owes their split amount
            friend_split = next((sp for sp in exp.splits if sp.user_id == friend_id), None)
            if friend_split:
                impact = friend_split.owed_amount
                net_balance += impact
                timeline.append(
                    ActivityTimelineItem(
                        type="EXPENSE",
                        id=exp.id,
                        date=exp.expense_date,
                        description=exp.description,
                        total_amount=exp.amount,
                        currency=exp.currency,
                        payer_id=exp.payer_id,
                        payer_name=current_user.name,
                        group_id=exp.group_id,
                        your_share=friend_split.owed_amount,
                        your_net_impact=impact,
                    )
                )
        elif exp.payer_id == friend_id:
            # Friend paid: caller owes their split amount
            caller_split = next((sp for sp in exp.splits if sp.user_id == current_user.id), None)
            if caller_split:
                impact = -caller_split.owed_amount
                net_balance += impact
                timeline.append(
                    ActivityTimelineItem(
                        type="EXPENSE",
                        id=exp.id,
                        date=exp.expense_date,
                        description=exp.description,
                        total_amount=exp.amount,
                        currency=exp.currency,
                        payer_id=exp.payer_id,
                        payer_name=friend.name,
                        group_id=exp.group_id,
                        your_share=caller_split.owed_amount,
                        your_net_impact=impact,
                    )
                )

    for s in settlements:
        if s.payer_id == current_user.id:
            # Caller paid friend: reduces debt to friend (increases net_balance)
            net_balance += s.amount
            timeline.append(
                ActivityTimelineItem(
                    type="SETTLEMENT",
                    id=s.id,
                    date=s.settlement_date,
                    description=f"Paid {friend.name} via {s.payment_method}",
                    total_amount=s.amount,
                    currency=s.currency,
                    payer_id=s.payer_id,
                    payer_name=current_user.name,
                    payee_id=s.payee_id,
                    payee_name=friend.name,
                    group_id=s.group_id,
                    your_net_impact=s.amount,
                )
            )
        else:
            # Friend paid caller: friend reduced debt to caller
            net_balance -= s.amount
            timeline.append(
                ActivityTimelineItem(
                    type="SETTLEMENT",
                    id=s.id,
                    date=s.settlement_date,
                    description=f"{friend.name} paid you via {s.payment_method}",
                    total_amount=s.amount,
                    currency=s.currency,
                    payer_id=s.payer_id,
                    payer_name=friend.name,
                    payee_id=s.payee_id,
                    payee_name=current_user.name,
                    group_id=s.group_id,
                    your_net_impact=-s.amount,
                )
            )

    timeline.sort(key=lambda x: x.date, reverse=True)

    if net_balance > Decimal("0.00"):
        summary_text = f"{friend.name} owes you ₹{net_balance:.2f}"
    elif net_balance < Decimal("0.00"):
        summary_text = f"You owe {friend.name} ₹{abs(net_balance):.2f}"
    else:
        summary_text = f"You and {friend.name} are all settled up!"

    return BilateralActivityFeed(
        friend_id=friend.id,
        friend_name=friend.name,
        net_balance=net_balance,
        currency="INR",
        summary_text=summary_text,
        timeline=timeline[offset : offset + limit],
    )


# -------------------------------------------------------------
# BALANCES
# -------------------------------------------------------------


@router.get(
    "/balances",
    response_model=BalanceSummary,
    summary="Compute real-time net balances across verified expenses and settlements",
)
async def get_balances(
    group_id: uuid.UUID | None = Query(None, description="Optional group filter"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> BalanceSummary:
    """
    Calculate exact net balances for the active user against all friends.

    Positive balance indicates money others owe to the user.
    Negative balance indicates money the user owes to others.
    """
    return await get_user_balances(
        session=session,
        user_id=current_user.id,
        group_id=group_id,
    )
