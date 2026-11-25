"""Audit activity log models for tracking shared expense and settlement lifecycle events."""

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, Column, Index, Numeric
from sqlmodel import Field, Relationship, SQLModel


class ActivityLog(SQLModel, table=True):
    """Immutable audit log entry for expense and settlement actions."""

    __tablename__ = "activity_logs"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    event_type: str = Field(
        max_length=50,
        index=True,
        description="EXPENSE_CREATED, EXPENSE_UPDATED, EXPENSE_DELETED, SETTLEMENT_CREATED, SETTLEMENT_DELETED.",
    )
    entity_type: str = Field(
        max_length=30,
        index=True,
        description="EXPENSE or SETTLEMENT.",
    )
    entity_id: uuid.UUID = Field(index=True, nullable=False)
    actor_id: uuid.UUID = Field(foreign_key="users.id", index=True, nullable=False)
    group_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="groups.id",
        index=True,
        nullable=True,
    )
    description: str = Field(max_length=255)
    amount: Decimal = Field(
        sa_column=Column(Numeric(12, 2), nullable=False),
        description="Snapshot of the expense or settlement amount at the time of this event.",
    )
    currency: str = Field(default="INR", max_length=10)
    category: str | None = Field(default="general", max_length=100, nullable=True)
    payer_id: uuid.UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    payee_id: uuid.UUID | None = Field(default=None, foreign_key="users.id", nullable=True)
    metadata_json: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False, default=dict),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False,
        index=True,
    )

    audiences: list["ActivityLogAudience"] = Relationship(
        back_populates="activity",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

    __table_args__ = (
        Index("ix_activity_entity", "entity_type", "entity_id"),
        Index("ix_activity_group_created", "group_id", "created_at"),
    )


class ActivityLogAudience(SQLModel, table=True):
    """Per-user visibility mapping for an ActivityLog entry."""

    __tablename__ = "activity_log_audiences"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    activity_id: uuid.UUID = Field(foreign_key="activity_logs.id", index=True, nullable=False)
    user_id: uuid.UUID = Field(foreign_key="users.id", index=True, nullable=False)

    activity: ActivityLog | None = Relationship(back_populates="audiences")

    __table_args__ = (
        Index("ix_activity_audience_user_activity", "user_id", "activity_id", unique=True),
    )
