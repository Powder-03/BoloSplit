"""Schemas for the Staged Expenses review stack and batch approvals."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StagedExpenseRead(BaseModel):
    """Client representation of a staged expense review card."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    group_id: uuid.UUID | None = None
    action_type: str
    status: str
    target_expense_id: uuid.UUID | None = None
    description: str
    amount: Decimal
    currency: str
    payer_id: uuid.UUID | None = None
    payer_name: str | None = None
    split_data: list[dict[str, Any]] = Field(default_factory=list)
    expense_date: datetime
    category: str | None = "general"
    confidence: str
    candidate_options: list[dict[str, Any]] | None = None
    is_possible_duplicate: bool
    raw_voice_transcript: str | None = None
    spoken_confirmation: str | None = None
    created_at: datetime
    updated_at: datetime



class StagedExpenseUpdate(BaseModel):
    """Payload to patch fields in a staged review item before approving."""

    description: str | None = None
    amount: Decimal | None = None
    category: str | None = None
    payer_id: uuid.UUID | None = None
    payer_name: str | None = None
    group_id: uuid.UUID | None = None
    expense_date: datetime | None = None
    split_data: list[dict[str, Any]] | None = None


class BatchApproveRequest(BaseModel):
    """Optional list of specific staged IDs to approve; if empty, approves all pending."""

    staged_ids: list[uuid.UUID] | None = Field(
        default=None,
        description="Optional list of specific staged item IDs to approve. If omitted, all PENDING items for the user/group are approved.",
    )


class BatchApproveResponse(BaseModel):
    """Response returned upon approving staged items atomically."""

    approved_count: int
    rejected_count: int = 0
    approved_expense_ids: list[uuid.UUID] = Field(default_factory=list)
    message: str = "Staged actions successfully promoted to the ledger."
