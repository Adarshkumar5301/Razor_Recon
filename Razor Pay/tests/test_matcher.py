"""
tests/test_matcher.py
=======================

Unhappy-path coverage for the deterministic matcher. The matcher's entire
job is to be strict, so most of these tests exist to prove it correctly
REFUSES to match things a naive implementation might accidentally clear.
"""
import json
from pathlib import Path

import pytest

from src.deterministic.matcher import run_deterministic_match


@pytest.fixture()
def tmp_dataset(tmp_path: Path):
    """Write a small, hand-crafted merchant_sales.json / razorpay_payouts.json
    pair covering every unhappy path, and return their paths."""

    sales = [
        {"order_id": "ORD-A", "timestamp": "2026-01-01T00:00:00Z", "amount": 1000.00, "customer_id": "C-1"},
        {"order_id": "ORD-B", "timestamp": "2026-01-01T00:00:00Z", "amount": 500.00, "customer_id": "C-2"},
        {"order_id": "ORD-C", "timestamp": "2026-01-01T00:00:00Z", "amount": 250.00, "customer_id": "C-3"},
        {"order_id": "ORD-D", "timestamp": "2026-01-01T00:00:00Z", "amount": 100.50, "customer_id": "C-4"},
        # ORD-E deliberately has NO payout at all.
        {"order_id": "ORD-E", "timestamp": "2026-01-01T00:00:00Z", "amount": 75.00, "customer_id": "C-5"},
    ]

    payouts = [
        # Clean 1:1 exact match.
        {"payout_id": "PO-A", "linked_orders": ["ORD-A"], "settled_amount": 1000.00,
         "gateway_fee_deducted": 0.0, "tax_deducted": 0.0},
        # Near-miss by a single paisa — must NOT match (zero tolerance).
        {"payout_id": "PO-B", "linked_orders": ["ORD-B"], "settled_amount": 499.99,
         "gateway_fee_deducted": None, "tax_deducted": None},
        # Bundled payout referencing two orders whose sum matches exactly —
        # must still be UNRESOLVED because it isn't a strict 1:1 record.
        {"payout_id": "PO-C", "linked_orders": ["ORD-C", "ORD-D"], "settled_amount": 350.50,
         "gateway_fee_deducted": 0.0, "tax_deducted": 0.0},
        # References an order id that doesn't exist in the merchant ledger.
        {"payout_id": "PO-D", "linked_orders": ["ORD-GHOST"], "settled_amount": 999.00,
         "gateway_fee_deducted": 0.0, "tax_deducted": 0.0},
    ]

    sales_path = tmp_path / "merchant_sales.json"
    payouts_path = tmp_path / "razorpay_payouts.json"
    sales_path.write_text(json.dumps(sales), encoding="utf-8")
    payouts_path.write_text(json.dumps(payouts), encoding="utf-8")
    return sales_path, payouts_path


def test_exact_match_clears(tmp_dataset):
    sales_path, payouts_path = tmp_dataset
    result = run_deterministic_match(sales_path, payouts_path)
    matched_payout_ids = {m.payout_id for m in result.matched}
    assert "PO-A" in matched_payout_ids


def test_paisa_level_mismatch_is_not_matched(tmp_dataset):
    """A 1-paisa difference must NOT be silently cleared — zero tolerance by design."""
    sales_path, payouts_path = tmp_dataset
    result = run_deterministic_match(sales_path, payouts_path)
    matched_payout_ids = {m.payout_id for m in result.matched}
    unresolved_payout_ids = {u.payout_id for u in result.unresolved}
    assert "PO-B" not in matched_payout_ids
    assert "PO-B" in unresolved_payout_ids
    reason = next(u.reason for u in result.unresolved if u.payout_id == "PO-B")
    assert reason == "AMOUNT_MISMATCH"


def test_bundled_payout_never_deterministically_matched_even_if_sum_is_exact(tmp_dataset):
    """This is the key correctness guarantee of Phase 3: even a PERFECT sum
    match across multiple orders must be routed to the AI layer, because the
    deterministic layer's contract is strict 1:1 only."""
    sales_path, payouts_path = tmp_dataset
    result = run_deterministic_match(sales_path, payouts_path)
    matched_payout_ids = {m.payout_id for m in result.matched}
    unresolved = {u.payout_id: u for u in result.unresolved}
    assert "PO-C" not in matched_payout_ids
    assert "PO-C" in unresolved
    assert unresolved["PO-C"].reason == "BUNDLED_MULTI_ORDER_PAYOUT"
    # Sanity: the unresolved record still carries the correct combined total
    # for the AI layer to later reconcile.
    assert unresolved["PO-C"].merchant_total == pytest.approx(350.50)


def test_unknown_order_reference_is_unresolved_not_crashed(tmp_dataset):
    sales_path, payouts_path = tmp_dataset
    result = run_deterministic_match(sales_path, payouts_path)
    unresolved = {u.payout_id: u for u in result.unresolved}
    assert "PO-D" in unresolved
    assert unresolved["PO-D"].reason == "ORDER_NOT_FOUND_IN_MERCHANT_LEDGER"
    # No merchant amount is known for the ghost order, so the map is empty.
    assert unresolved["PO-D"].merchant_amounts == {}


def test_order_with_no_payout_at_all_is_surfaced(tmp_dataset):
    sales_path, payouts_path = tmp_dataset
    result = run_deterministic_match(sales_path, payouts_path)
    no_payout = [u for u in result.unresolved if u.payout_id == "NO_PAYOUT_FOUND:ORD-E" and "ORD-E" in u.linked_orders]
    assert len(no_payout) == 1
    assert no_payout[0].reason == "NO_PAYOUT_FOUND_FOR_ORDER"


def test_totals_are_internally_consistent(tmp_dataset):
    sales_path, payouts_path = tmp_dataset
    result = run_deterministic_match(sales_path, payouts_path)
    assert result.total_merchant_orders == 5
    assert result.total_payouts == 4
    # 1 matched (PO-A) + 3 unresolved-from-payouts (PO-B, PO-C, PO-D) + 1 no-payout (ORD-E)
    assert len(result.matched) == 1
    assert len(result.unresolved) == 4


def test_duplicate_linked_orders_rejected_by_schema(tmp_path):
    """Pydantic validation must reject a payout that lists the same order twice —
    that's malformed input, not a legitimate bundled settlement."""
    sales_path = tmp_path / "merchant_sales.json"
    payouts_path = tmp_path / "razorpay_payouts.json"
    sales_path.write_text(json.dumps([
        {"order_id": "ORD-X", "timestamp": "2026-01-01T00:00:00Z", "amount": 10.0, "customer_id": "C-1"}
    ]), encoding="utf-8")
    payouts_path.write_text(json.dumps([
        {"payout_id": "PO-X", "linked_orders": ["ORD-X", "ORD-X"], "settled_amount": 20.0,
         "gateway_fee_deducted": 0.0, "tax_deducted": 0.0}
    ]), encoding="utf-8")

    with pytest.raises(Exception):
        run_deterministic_match(sales_path, payouts_path)


def test_order_referenced_by_multiple_payouts_is_never_auto_matched(tmp_path):
    """Prevent a duplicate settlement file row from double-counting revenue."""
    sales_path = tmp_path / "merchant_sales.json"
    payouts_path = tmp_path / "razorpay_payouts.json"
    sales_path.write_text(json.dumps([
        {"order_id": "ORD-X", "timestamp": "2026-01-01T00:00:00Z", "amount": 10.0, "customer_id": "C-1"}
    ]), encoding="utf-8")
    payouts_path.write_text(json.dumps([
        {"payout_id": "PO-X1", "linked_orders": ["ORD-X"], "settled_amount": 10.0,
         "gateway_fee_deducted": 0.0, "tax_deducted": 0.0},
        {"payout_id": "PO-X2", "linked_orders": ["ORD-X"], "settled_amount": 10.0,
         "gateway_fee_deducted": 0.0, "tax_deducted": 0.0}
    ]), encoding="utf-8")

    result = run_deterministic_match(sales_path, payouts_path)
    assert result.matched == []
    assert {record.reason for record in result.unresolved} == {"ORDER_APPEARS_IN_MULTIPLE_PAYOUTS"}
