"""Staging queue model for voice-first human-in-the-loop review."""

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, Column, Numeric
from sqlmodel import Field, SQLModel


class StagedExpense(SQLModel, table=True):
    """Voice-parsed action review card isolating voice input from verified ledger."""

    __tablename__ = "staged_expenses"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    user_id: uuid.UUID = Field(foreign_key="users.id", index=True, nullable=False)
    group_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="groups.id",
        index=True,
        nullable=True,
    )
    action_type: str = Field(
        default="CREATE",
        index=True,
        max_length=20,
        description="Action to perform upon approval: CREATE, UPDATE, or DELETE.",
    )
    status: str = Field(
        default="PENDING",
        index=True,
        max_length=20,
        description="Queue status: PENDING, APPROVED, or REJECTED.",
    )
    target_expense_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="expenses.id",
        index=True,
        nullable=True,
        description="Pointer to target Expense for UPDATE or DELETE actions.",
    )
    description: str = Field(max_length=255, index=True)
    amount: Decimal = Field(
        sa_column=Column(Numeric(12, 2), nullable=False),
        description="Parsed expense amount in Decimal with exact paise precision.",
    )
    currency: str = Field(default="INR", max_length=10)
    payer_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="users.id",
        nullable=True,
    )
    payer_name: str | None = Field(
        default=None,
        max_length=100,
        description="Raw extracted payer name or phrase (e.g. 'me', 'Sahaj').",
    )
    split_data: list[dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False, default=list),
        description="Resolved or draft participant splits: [{user_id, raw_name, owed_amount}].",
    )
    expense_date: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False,
        index=True,
    )
    category: str | None = Field(default="general", max_length=100, index=True)
    confidence: str = Field(
        default="HIGH",
        max_length=20,
        description="AI confidence: HIGH, MEDIUM, or AMBIGUOUS.",
    )
    candidate_options: list[dict[str, Any]] | None = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
        description="List of conflicting candidate expenses when confidence is AMBIGUOUS.",
    )
    is_possible_duplicate: bool = Field(
        default=False,
        index=True,
        description="Flagged true if identical amount+category staged in the last 10 minutes.",
    )
    raw_voice_transcript: str | None = Field(
        default=None,
        description="The original user voice transcript or text query.",
    )
    spoken_confirmation: str | None = Field(
        default=None,
        description="Generated Hinglish conversational confirmation.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False,
        index=True,
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False,
    )
