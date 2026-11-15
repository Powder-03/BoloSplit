"""Schemas for verified ledger, groups, friends, settlements, and activity feeds."""

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class ExpenseSplitCreate(BaseModel):
    """Payload to specify a split share."""

    user_id: uuid.UUID
    owed_amount: Decimal
    split_type: str = "EQUAL"
    split_value: Decimal | None = None


class ExpenseSplitRead(BaseModel):
    """Split breakdown item in response."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    expense_id: uuid.UUID
    user_id: uuid.UUID
    owed_amount: Decimal
    split_type: str
    split_value: Decimal | None = None
    created_at: datetime


class ExpenseCreate(BaseModel):
    """Manual fallback creation payload for an expense."""

    description: str
    amount: Decimal
    currency: str = "INR"
    payer_id: uuid.UUID
    group_id: uuid.UUID | None = None
    category: str | None = "general"
    expense_date: datetime | None = None
    splits: list[ExpenseSplitCreate] = Field(default_factory=list)


class ExpenseUpdate(BaseModel):
    """Payload to update an expense."""

    description: str | None = None
    amount: Decimal | None = None
    category: str | None = None
    expense_date: datetime | None = None


class ExpenseRead(BaseModel):
    """Verified ledger expense item."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    group_id: uuid.UUID | None = None
    description: str
    amount: Decimal
    currency: str
    payer_id: uuid.UUID
    payer_name: str | None = None
    category: str | None = "general"
    expense_date: datetime
    is_deleted: bool
    created_at: datetime
    updated_at: datetime
    splits: list[ExpenseSplitRead] = Field(default_factory=list)


class GroupCreate(BaseModel):
    """Payload to create a new group."""

    name: str = Field(..., max_length=255, description="Group name (e.g. 'Flat 302', 'Goa Trip').")


class GroupRead(BaseModel):
    """Group details response."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    invite_code: str
    created_by: uuid.UUID
    created_at: datetime


class GroupMemberRead(BaseModel):
    """Member inside a group."""

    user_id: uuid.UUID
    name: str
    email: str
    phone: str | None = None
    role: str = "member"
    joined_at: datetime


class GroupDetailRead(BaseModel):
    """Detailed group profile with member list and total spending."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    invite_code: str
    created_by: uuid.UUID
    created_at: datetime
    members: list[GroupMemberRead] = Field(default_factory=list)
    total_spending: Decimal = Decimal("0.00")


class GroupMemberAdd(BaseModel):
    """Payload to add a member directly to a group by email, phone, or UUID."""

    identifier: str = Field(..., description="User UUID, email, or phone number.")
    name: str | None = Field(default=None, description="Optional name if auto-provisioning new user.")
    role: str = Field(default="member", description="admin or member.")


class FriendCreate(BaseModel):
    """Payload to register a friend with nicknames and spoken aliases."""

    friend_email: str | None = None
    friend_phone: str | None = None
    friend_name: str | None = None
    nickname: str | None = None
    aliases: list[str] = Field(default_factory=list)


class FriendRead(BaseModel):
    """Friendship details."""

    id: uuid.UUID
    user_id: uuid.UUID
    friend_id: uuid.UUID
    friend_name: str
    friend_email: str
    friend_phone: str | None = None
    nickname: str | None = None
    aliases: list[str] = Field(default_factory=list)
    created_at: datetime


class UserBalance(BaseModel):
    """Net balance with a specific user."""

    user_id: uuid.UUID
    user_name: str
    net_balance: Decimal = Field(
        ...,
        description="Positive if they owe you, negative if you owe them.",
    )


class BalanceSummary(BaseModel):
    """Overall and per-friend balance overview."""

    total_owed_to_you: Decimal = Field(
        default=Decimal("0.00"),
        description="Total money others owe to you.",
    )
    total_you_owe: Decimal = Field(
        default=Decimal("0.00"),
        description="Total money you owe to others.",
    )
    net_total: Decimal = Field(
        default=Decimal("0.00"),
        description="Net overall position.",
    )
    balances: list[UserBalance] = Field(default_factory=list)


class ActivityTimelineItem(BaseModel):
    """Unified item in shared activity feed (Expense or Settlement)."""

    type: str  # "EXPENSE" or "SETTLEMENT"
    id: uuid.UUID
    date: datetime
    description: str
    total_amount: Decimal
    currency: str = "INR"
    payer_id: uuid.UUID
    payer_name: str
    payee_id: uuid.UUID | None = None
    payee_name: str | None = None
    group_id: uuid.UUID | None = None
    your_share: Decimal | None = None
    your_net_impact: Decimal = Field(
        ...,
        description="Positive if this transaction increased what you are owed or paid off debt; negative if it increased what you owe.",
    )


class BilateralActivityFeed(BaseModel):
    """Shared 1-on-1 activity feed between active user and a friend."""

    friend_id: uuid.UUID
    friend_name: str
    net_balance: Decimal
    currency: str = "INR"
    summary_text: str
    timeline: list[ActivityTimelineItem] = Field(default_factory=list)


class GroupActivityFeed(BaseModel):
    """Shared group activity feed with timeline of expenses and settlements."""

    group_id: uuid.UUID
    group_name: str
    total_spending: Decimal
    currency: str = "INR"
    timeline: list[ActivityTimelineItem] = Field(default_factory=list)
