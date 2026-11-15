"""Unit tests verifying exact paise split arithmetic invariants (Equal and Percentage)."""

import uuid
from decimal import Decimal

import pytest

from app.services.ledger_service import calculate_equal_splits, calculate_percentage_splits


@pytest.mark.parametrize(
    "amount_str,n_participants",
    [
        ("100.00", 3),
        ("1000.00", 6),
        ("10.00", 7),
        ("0.05", 4),
        ("500.00", 2),
        ("237.89", 5),
        ("0.01", 3),
        ("9999.99", 11),
    ],
)
def test_exact_paise_distribution_invariant(amount_str: str, n_participants: int):
    """
    Verify that for any currency amount and any N participants,
    the sum of individual decimal splits exactly equals the original amount.
    Zero floating point drift or lost paise allowed.
    """
    amount = Decimal(amount_str)
    participants = [uuid.uuid4() for _ in range(n_participants)]
    payer_id = participants[0]

    splits = calculate_equal_splits(
        amount=amount,
        participant_ids=participants,
        payer_id=payer_id,
    )

    assert len(splits) == n_participants
    split_sum = sum(owed for _, owed in splits)
    assert split_sum == amount, f"Split sum {split_sum} must equal total {amount}"

    for _, owed in splits:
        assert owed >= Decimal("0.00")


@pytest.mark.parametrize(
    "amount_str,percentages",
    [
        ("500.00", [("20.00", "me"), ("40.00", "sahaj"), ("40.00", "alex")]),
        ("100.00", [("33.33", "u1"), ("33.33", "u2"), ("33.34", "u3")]),
        ("1000.00", [("50.00", "u1"), ("25.00", "u2"), ("25.00", "u3")]),
        ("123.45", [("10.00", "u1"), ("20.00", "u2"), ("30.00", "u3"), ("40.00", "u4")]),
        ("0.05", [("50.00", "u1"), ("50.00", "u2")]),
        ("999.99", [("14.28", "u1"), ("14.28", "u2"), ("14.28", "u3"), ("14.28", "u4"), ("14.28", "u5"), ("14.30", "u6"), ("14.30", "u7")]),
    ],
)
def test_percentage_splits_paise_invariant(amount_str: str, percentages: list[tuple[str, str]]):
    """
    Verify that Hare-Niemeyer percentage split allocates exact paise with zero loss.
    """
    amount = Decimal(amount_str)
    user_ids = [uuid.uuid4() for _ in range(len(percentages))]
    pct_tuples = [(user_ids[i], Decimal(pct)) for i, (pct, _) in enumerate(percentages)]
    payer_id = user_ids[0]

    splits = calculate_percentage_splits(
        amount=amount,
        participant_percentages=pct_tuples,
        payer_id=payer_id,
    )

    assert len(splits) == len(percentages)
    split_sum = sum(owed for _, owed in splits)
    assert split_sum == amount, f"Percentage split sum {split_sum} must equal total {amount}"

    # For the specific Goa Trip test case: 500 split 20%, 40%, 40% -> 100, 200, 200
    if amount_str == "500.00" and len(percentages) == 3:
        assert splits[0][1] == Decimal("100.00")
        assert splits[1][1] == Decimal("200.00")
        assert splits[2][1] == Decimal("200.00")
