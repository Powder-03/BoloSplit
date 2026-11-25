"""Pydantic schemas for single-pass multimodal AI extraction and ledger resolution."""

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field


class IntentType(StrEnum):
    """Supported voice intent types."""

    CREATE = "CREATE"
    MULTI_CREATE = "MULTI_CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    SETTLE = "SETTLE"
    GET = "GET"
    UNKNOWN = "UNKNOWN"


class ExpenseScope(StrEnum):
    """Explicit separation between personal (SELF) and friend/group (SHARED) expenses."""

    SELF = "SELF"
    SHARED = "SHARED"


class SplitTypeHint(StrEnum):
    """Split method classification."""

    EQUAL = "EQUAL"
    EXACT = "EXACT"
    PERCENTAGE = "PERCENTAGE"
    SHARES = "SHARES"
    UNSPECIFIED = "UNSPECIFIED"


class PercentageSplitItem(BaseModel):
    """Percentage, exact amount, or share weight assignment for a single participant."""

    participant_raw: str = Field(..., description="Spoken name, alias, or UUID (e.g. 'me', 'Sahaj').")
    percentage: float = Field(..., description="Percentage (0-100), exact amount in INR, or share weight.")


class ExactSplitItem(BaseModel):
    """Exact INR amount assignment for a single participant."""

    participant_raw: str = Field(..., description="Spoken name, alias, or UUID (e.g. 'me', 'Sahaj').")
    amount: float = Field(..., description="Exact amount owed by this participant in INR.")


class ShareSplitItem(BaseModel):
    """Relative share weight assignment for a single participant."""

    participant_raw: str = Field(..., description="Spoken name, alias, or UUID (e.g. 'me', 'Sahaj').")
    shares: float = Field(..., description="Relative share weight (e.g. 1.0, 2.0).")


class ExtractedItem(BaseModel):
    """Single expense item extracted from voice audio or text in a single pass."""

    description: str = Field(
        ...,
        description="Clean title/description of the expense (e.g., 'Petrol', 'Dinner at Sagar Ratna').",
    )
    amount: float = Field(
        ...,
        description="Numeric total normalized to INR (e.g., 'dedh hazar' -> 1500.0, '350' -> 350.0).",
    )
    currency: str = Field(default="INR", description="Currency ISO code.")
    scope: ExpenseScope = Field(
        default=ExpenseScope.SHARED,
        description="SELF for strictly personal expenses (no friends/groups); SHARED for friends or groups.",
    )
    payer_raw: str | None = Field(
        default="me",
        description="Who paid: 'me' if caller paid, or spoken friend name/nickname if a friend paid.",
    )
    payer_user_id: str | None = Field(
        default=None,
        description="Optional matched user_id from Known Directory if resolved directly.",
    )
    participants_raw: list[str] = Field(
        default_factory=list,
        description="List of spoken participant names/nicknames (e.g. ['Sahaj'], ['me', 'Sahaj']).",
    )
    group_raw: str | None = Field(
        default=None,
        description="Spoken group name if mentioned (e.g. 'Goa Trip', 'Flatmates').",
    )
    group_id: str | None = Field(
        default=None,
        description="Optional matched group_id UUID from Known Groups.",
    )
    category: str | None = Field(
        default="general",
        description="Classified category (e.g., 'food', 'travel', 'fuel', 'groceries', 'utilities').",
    )
    expense_date_iso: str | None = Field(
        default=None,
        description="Normalized date in YYYY-MM-DD or ISO 8601 format anchored to IST.",
    )
    split_type: SplitTypeHint = Field(
        default=SplitTypeHint.EQUAL,
        description="Detected split type: EQUAL, PERCENTAGE, EXACT, or SHARES.",
    )
    percentage_splits: list[PercentageSplitItem] = Field(
        default_factory=list,
        description="List of percentage assignments if split_type is PERCENTAGE.",
    )
    exact_splits: list[ExactSplitItem] = Field(
        default_factory=list,
        description="List of exact currency assignments if split_type is EXACT.",
    )
    share_splits: list[ShareSplitItem] = Field(
        default_factory=list,
        description="List of share ratio assignments if split_type is SHARES.",
    )
    split_details_raw: str | None = Field(
        default=None,
        description="Optional raw split note.",
    )


class SettleExtraction(BaseModel):
    """Details for a bidirectional debt settlement command ('I paid Sahaj 250' or 'Sahaj paid me 250')."""

    payer_raw: str = Field(
        default="me",
        description="Person who paid the money ('me' if caller paid friend; friend's name if friend paid caller).",
    )
    payee_raw: str = Field(
        ...,
        description="Person who received the money (friend's name if caller paid friend; 'me' if friend paid caller).",
    )
    amount: float = Field(..., description="Amount paid in INR.")
    currency: str = Field(default="INR")
    payment_method: str = Field(default="UPI", description="UPI, CASH, GPAY, etc.")
    group_raw: str | None = Field(default=None, description="Group name if settling in a group.")
    notes: str | None = Field(default=None)


class CandidateSearchQuery(BaseModel):
    """Parameters to search or disambiguate existing expenses for UPDATE or DELETE."""

    description_keywords: list[str] = Field(
        default_factory=list,
        description="Single-word core keywords to match expense descriptions (e.g. ['petrol']).",
    )
    amount_hint: float | None = Field(
        default=None,
        description="Expected current amount or new amount if mentioned.",
    )
    date_target_iso: str | None = Field(
        default=None,
        description="Target date in YYYY-MM-DD (e.g. yesterday's date).",
    )
    date_window_days: int = Field(
        default=3,
        description="Search window in days around target date.",
    )


class StageOneOutput(BaseModel):
    """Unified single-pass structured output returned by Gemini multimodal perception."""

    intent: IntentType = Field(
        ...,
        description="Classified intent of the voice utterance.",
    )
    confidence: str = Field(
        default="HIGH",
        description="Extraction confidence: HIGH, MEDIUM, LOW, or AMBIGUOUS.",
    )
    ambiguity_reason: str | None = Field(
        default=None,
        description="Explanation when confidence is MEDIUM, LOW, or AMBIGUOUS.",
    )
    is_complex: bool = Field(
        default=False,
        description="Backward-compatible flag.",
    )
    complexity_reason: str | None = Field(
        default=None,
        description="Backward-compatible note.",
    )
    items: list[ExtractedItem] = Field(
        default_factory=list,
        description="List of parsed expense items (for CREATE, MULTI_CREATE, or UPDATE new values).",
    )
    settle_item: SettleExtraction | None = Field(
        default=None,
        description="Extracted settlement payment details if intent is SETTLE.",
    )
    search_query: CandidateSearchQuery | None = Field(
        default=None,
        description="Search parameters for UPDATE or DELETE.",
    )
    target_record_id: str | None = Field(
        default=None,
        description="For UPDATE/DELETE: matched expense ID from Recent Expenses Context when unambiguous.",
    )
    ambiguous_candidate_ids: list[str] = Field(
        default_factory=list,
        description="For UPDATE/DELETE: list of conflicting expense IDs when 2+ records match.",
    )
    clarification_question: str | None = Field(
        default=None,
        description="Question to ask user when multiple records or friends match.",
    )
    spoken_confirmation: str | None = Field(
        default=None,
        description="Conversational Hinglish/English confirmation summary.",
    )


# Backward-compatible aliases for tests and external callers
VoiceExtractionOutput = StageOneOutput


class ResolvedSplit(BaseModel):
    """Backward-compatible resolved split item."""

    user_id: str | None = None
    raw_name: str
    owed_amount: Decimal
    is_payer: bool = False


class StageTwoResolution(BaseModel):
    """Backward-compatible resolution model."""

    action_type: str = "UPDATE"
    group_id: str | None = None
    target_record_id: str | None = None
    target_source: str | None = "verified"
    confidence: str = "HIGH"
    ambiguous_candidate_ids: list[str] = Field(default_factory=list)
    description: str = ""
    amount: Decimal = Decimal("0.00")
    currency: str = "INR"
    payer_user_id: str | None = None
    payer_name: str | None = None
    payee_user_id: str | None = None
    payee_name: str | None = None
    resolved_splits: list[ResolvedSplit] = Field(default_factory=list)
    expense_date_iso: str | None = None
    category: str | None = "general"
    spoken_confirmation: str | None = None
    clarification_question: str | None = None
