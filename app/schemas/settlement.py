"""Schemas for Settlements ('Settle Up' debt payments)."""

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class SettlementCreate(BaseModel):
    """Payload to record a settlement payment (supports both outgoing and incoming payments)."""

    payer_id: uuid.UUID | None = Field(
        default=None,
        description="User ID of person who paid (defaults to current_user if omitted; set to friend's ID when friend paid you).",
    )
    payee_id: uuid.UUID = Field(..., description="User ID of person receiving the money.")
    amount: Decimal = Field(..., gt=0, description="Amount paid in decimal.")
    currency: str = Field(default="INR", max_length=10)
    group_id: uuid.UUID | None = Field(default=None, description="Optional group context.")
    payment_method: str = Field(
        default="UPI",
        description="Payment rail: UPI, GPAY, PHONEPE, CASH, NET_BANKING, OTHER.",
    )
    transaction_reference: str | None = Field(
        default=None,
        description="UTR, UPI reference ID, or bank reference.",
    )
    notes: str | None = Field(default=None, description="Optional note (e.g. 'Paid via GPay').")
    settlement_date: datetime | None = Field(
        default=None,
        description="Payment timestamp (defaults to current time).",
    )


class SettlementRead(BaseModel):
    """Settlement record details."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    payer_id: uuid.UUID
    payer_name: str | None = None
    payee_id: uuid.UUID
    payee_name: str | None = None
    amount: Decimal
    currency: str
    group_id: uuid.UUID | None = None
    payment_method: str
    transaction_reference: str | None = None
    notes: str | None = None
    settlement_date: datetime
    is_deleted: bool
    created_at: datetime
