"""Schemas for voice command ingestion (Direct-to-Ledger)."""

import uuid
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.ledger import BalanceSummary, ExpenseRead, SelfExpenseRead
from app.schemas.settlement import SettlementRead


class VoiceCommandRequest(BaseModel):
    """Voice text ingestion request payload."""

    text: str = Field(
        ...,
        description="Transcribed voice command (e.g. 'Add 500 rupees for petrol split with Sahaj').",
    )
    client_timestamp_iso: str | None = Field(
        default=None,
        description="Client local timestamp in ISO format (defaults to current Asia/Kolkata IST).",
    )
    group_id: uuid.UUID | None = Field(
        default=None,
        description="Optional group ID if command was issued inside a group view.",
    )


class VoiceIngestResponse(BaseModel):
    """Response returned upon directly executing a voice command on the ledger."""

    status: str = Field(
        default="SUCCESS",
        description="'SUCCESS', 'AMBIGUOUS', 'ERROR', etc.",
    )
    action_type: str = Field(
        default="CREATE",
        description="Action performed: CREATE, MULTI_CREATE, UPDATE, DELETE, SETTLE, GET.",
    )
    count: int = Field(default=1, description="Number of items created or modified.")
    expenses: list[ExpenseRead] = Field(
        default_factory=list,
        description="Created or updated shared ledger expenses.",
    )
    self_expenses: list[SelfExpenseRead] = Field(
        default_factory=list,
        description="Created or updated personal (self) expenses.",
    )
    settlement: SettlementRead | None = Field(
        default=None,
        description="Created settlement record if action was SETTLE.",
    )
    balance_summary: BalanceSummary | None = Field(
        default=None,
        description="Balance summary if action was GET.",
    )
    spoken_confirmation: str | None = Field(
        default=None,
        description="Conversational Hinglish confirmation for mobile TTS/audio reply.",
    )
    is_ambiguous: bool = Field(
        default=False,
        description="True if multiple matching candidates require user selection.",
    )
    clarification_prompt: str | None = Field(
        default=None,
        description="Clarification prompt for user if confidence is AMBIGUOUS or ERROR.",
    )
    error_code: str | None = Field(
        default=None,
        description="Machine-readable error code when status is ERROR or AMBIGUOUS.",
    )
    staged_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Backward-compatible empty list for legacy clients.",
    )
