"""Pydantic schemas for AI extraction, intent classification, and resolution."""

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


class SplitTypeHint(StrEnum):
    """Split method classification."""

    EQUAL = "EQUAL"
    EXACT = "EXACT"
    PERCENTAGE = "PERCENTAGE"
    SHARES = "SHARES"
    UNSPECIFIED = "UNSPECIFIED"


class PercentageSplitItem(BaseModel):
    """Percentage split assignment for a single participant."""

    participant_raw: str = Field(..., description="Spoken name or alias (e.g. 'me', 'Sahaj').")
    percentage: float = Field(..., description="Percentage number (e.g. 20.0, 40.0).")


class ExtractedItem(BaseModel):
    """Single expense item extracted from voice or transcript in Stage 1."""

    description: str = Field(
        ...,
        description="Clean title/description of the expense (e.g., 'Petrol', 'Dinner at Sagar Ratna').",
    )
    amount: float = Field(
        ...,
        description="Numeric total normalized to INR (e.g., 'dedh hazar' -> 1500.0, '350' -> 350.0).",
    )
    currency: str = Field(default="INR", description="Currency ISO code.")
    payer_raw: str | None = Field(
        default=None,
        description="Spoken payer expression (e.g. 'me', 'mine', 'Sahaj', 'Rohit').",
    )
    participants_raw: list[str] = Field(
        default_factory=list,
        description="List of spoken participant names or expressions (e.g. ['Sahaj'], ['me', 'Sahaj']).",
    )
    group_raw: str | None = Field(
        default=None,
        description="Raw spoken group name if mentioned (e.g. 'Goa Trip', 'Flatmates').",
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
        description="Detected split type hint.",
    )
    percentage_splits: list[PercentageSplitItem] = Field(
        default_factory=list,
        description="List of percentage assignments if split_type is PERCENTAGE.",
    )
    split_details_raw: str | None = Field(
        default=None,
        description="Any specific raw split details mentioned (e.g., '60-40', 'Sahaj gave 200').",
    )


class SettleExtraction(BaseModel):
    """Details for a debt settlement command ('I paid Sahaj 250 via GPay')."""

    payee_raw: str = Field(..., description="Person who received the money (e.g. 'Sahaj').")
    amount: float = Field(..., description="Amount paid.")
    currency: str = Field(default="INR")
    payment_method: str = Field(default="UPI", description="UPI, CASH, GPAY, etc.")
    group_raw: str | None = Field(default=None, description="Group name if settling in a group.")
    notes: str | None = Field(default=None)


class CandidateSearchQuery(BaseModel):
    """Parameters to search candidate expenses in DB for UPDATE/DELETE/complex actions."""

    description_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords to match expense descriptions (e.g. ['petrol', 'fuel']).",
    )
    amount_hint: float | None = Field(
        default=None,
        description="Expected amount or previous amount if mentioned.",
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
    """Structured output returned by Stage 1 Gemini 3.8 Flash model."""

    intent: IntentType = Field(
        ...,
        description="Classified intent of the voice utterance.",
    )
    confidence: str = Field(
        default="HIGH",
        description="Extraction confidence: HIGH, MEDIUM, or LOW.",
    )
    ambiguity_reason: str | None = Field(
        default=None,
        description="Explanation when confidence is MEDIUM or LOW.",
    )
    is_complex: bool = Field(
        default=False,
        description="True if relative past dates, UPDATE, DELETE, settlements, or unequal splits require Stage 2 reasoning.",
    )
    complexity_reason: str | None = Field(
        default=None,
        description="Explanation why this command was marked complex.",
    )
    items: list[ExtractedItem] = Field(
        default_factory=list,
        description="List of parsed expense items (supports MULTI_CREATE).",
    )
    settle_item: SettleExtraction | None = Field(
        default=None,
        description="Extracted settlement payment details if intent is SETTLE.",
    )
    search_query: CandidateSearchQuery | None = Field(
        default=None,
        description="Search query to fetch DB candidate records for Stage 2.",
    )
    spoken_confirmation: str | None = Field(
        default=None,
        description="Conversational Hinglish/English confirmation summary.",
    )


class ResolvedSplit(BaseModel):
    """Resolved split participant with exact owed amount."""

    user_id: str | None = Field(
        default=None,
        description="Resolved user UUID string, if mapped.",
    )
    raw_name: str = Field(
        ...,
        description="Participant spoken name or label.",
    )
    owed_amount: Decimal = Field(
        ...,
        description="Exact portion owed in Decimal paise.",
    )
    is_payer: bool = Field(
        default=False,
        description="Whether this participant paid.",
    )


class StageTwoResolution(BaseModel):
    """Structured resolution returned by Stage 2 Gemini 4 Hydrogen model."""

    action_type: str = Field(
        ...,
        description="Final action: CREATE, UPDATE, DELETE, SETTLE, or CLARIFICATION_REQUIRED.",
    )
    group_id: str | None = Field(
        default=None,
        description="Resolved Group UUID if expense or settlement belongs to a group.",
    )
    target_record_id: str | None = Field(
        default=None,
        description="Target record UUID if updating or deleting an existing expense.",
    )
    target_source: str | None = Field(
        default="verified",
        description="'verified' for expenses, 'settlements' for settlements.",
    )
    confidence: str = Field(
        default="HIGH",
        description="Confidence level: HIGH, MEDIUM, or AMBIGUOUS.",
    )
    ambiguous_candidate_ids: list[str] = Field(
        default_factory=list,
        description="IDs of conflicting candidates when 2+ matching records exist.",
    )
    description: str = Field(
        default="",
        description="Refined description of the expense or payment.",
    )
    amount: Decimal = Field(
        default=Decimal("0.00"),
        description="Total expense or settlement amount in Decimal.",
    )
    currency: str = Field(default="INR")
    payer_user_id: str | None = Field(
        default=None,
        description="UUID of payer.",
    )
    payer_name: str | None = Field(
        default=None,
        description="Name of payer.",
    )
    payee_user_id: str | None = Field(
        default=None,
        description="UUID of payee (for SETTLE).",
    )
    payee_name: str | None = Field(
        default=None,
        description="Name of payee (for SETTLE).",
    )
    resolved_splits: list[ResolvedSplit] = Field(
        default_factory=list,
        description="Exact splits summing to total amount.",
    )
    expense_date_iso: str | None = Field(
        default=None,
        description="Target expense date.",
    )
    category: str | None = Field(default="general")
    spoken_confirmation: str | None = Field(
        default=None,
        description="Friendly Hinglish reply for the user.",
    )
    clarification_question: str | None = Field(
        default=None,
        description="Question to ask if ambiguous candidates require clarification.",
    )
