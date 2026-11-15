"""Main verified ledger models: Expense and ExpenseSplit."""

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import Column, Numeric
from sqlmodel import Field, Relationship, SQLModel


class Expense(SQLModel, table=True):
    """Main verified expense ledger entry."""

    __tablename__ = "expenses"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    group_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="groups.id",
        index=True,
        nullable=True,
    )
    description: str = Field(max_length=255, index=True)
    amount: Decimal = Field(
        sa_column=Column(Numeric(12, 2), nullable=False),
        description="Total expense amount in Decimal with exact paise precision.",
    )
    currency: str = Field(default="INR", max_length=10)
    payer_id: uuid.UUID = Field(foreign_key="users.id", index=True, nullable=False)
    category: str | None = Field(default="general", max_length=100, index=True)
    expense_date: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False,
        index=True,
    )
    is_deleted: bool = Field(
        default=False,
        index=True,
        description="Soft delete flag for ledger audits.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relationships
    splits: list["ExpenseSplit"] = Relationship(
        back_populates="expense",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class ExpenseSplit(SQLModel, table=True):
    """Individual breakdown and owed share for an expense."""

    __tablename__ = "expense_splits"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    expense_id: uuid.UUID = Field(foreign_key="expenses.id", index=True, nullable=False)
    user_id: uuid.UUID = Field(foreign_key="users.id", index=True, nullable=False)
    owed_amount: Decimal = Field(
        sa_column=Column(Numeric(12, 2), nullable=False),
        description="Owed portion in Decimal (exact paise).",
    )
    split_type: str = Field(default="EQUAL", max_length=50)  # EQUAL, EXACT, PERCENTAGE
    split_value: Decimal | None = Field(
        default=None,
        sa_column=Column(Numeric(12, 2), nullable=True),
        description="Optional percentage or raw split input value.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relationships
    expense: Expense | None = Relationship(back_populates="splits")
