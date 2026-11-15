"""Settlement model for recording payments between users."""

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import Column, Index, Numeric
from sqlmodel import Field, SQLModel


class Settlement(SQLModel, table=True):
    """Represents a direct debt settlement (e.g. UPI, Cash) between two users."""

    __tablename__ = "settlements"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    payer_id: uuid.UUID = Field(
        foreign_key="users.id",
        index=True,
        nullable=False,
        description="The user sending the money.",
    )
    payee_id: uuid.UUID = Field(
        foreign_key="users.id",
        index=True,
        nullable=False,
        description="The user receiving the money.",
    )
    amount: Decimal = Field(
        sa_column=Column(Numeric(12, 2), nullable=False),
        description="Settlement amount in exact paise precision.",
    )
    currency: str = Field(default="INR", max_length=10)
    group_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="groups.id",
        index=True,
        nullable=True,
        description="Optional group association if settling inside a specific group.",
    )
    payment_method: str = Field(
        default="UPI",
        max_length=50,
        description="Payment rail: UPI, GPAY, PHONEPE, CASH, NET_BANKING, OTHER.",
    )
    transaction_reference: str | None = Field(
        default=None,
        max_length=100,
        nullable=True,
        description="Optional UTR / UPI reference number / notes.",
    )
    notes: str | None = Field(default=None, max_length=255, nullable=True)
    settlement_date: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False,
        index=True,
    )
    is_deleted: bool = Field(
        default=False,
        index=True,
        description="Soft delete flag for audit compliance and undo operations.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_settlement_payer_payee", "payer_id", "payee_id", "is_deleted"),
        Index("ix_settlement_group_date", "group_id", "settlement_date", "is_deleted"),
    )
