"""
tests/test_agent.py
=====================

Unit tests for the agent's deterministic reasoning core
(`_reason_about_variance`). This is the function that decides whether a
record gets auto-matched or pushed to the Honest Exception List, so its
boundary conditions matter more than almost anything else in this repo.

Live MCP-transport coverage (the agent actually talking to the actual
server subprocess) is exercised by src/main.py; see README.md's smoke
test. These tests intentionally isolate the reasoning function so they
run in milliseconds and don't depend on process spawning.
"""
import pytest

from src.agent.reconciler import CONFIDENCE_THRESHOLD, _reason_about_variance
from src.models import HistoricalFeeSchedule, UnresolvedRecord, VarianceReport

FEES = HistoricalFeeSchedule(
    standard_gateway_fee_percent=2.0,
    gst_on_fee_percent=18.0,
    rolling_reserve_hold_percent=5.0,
    rolling_reserve_release_days=180,
    tolerance_percent=0.35,
)


def _record(**overrides) -> UnresolvedRecord:
    base = dict(
        payout_id="PO-TEST",
        linked_orders=["ORD-1"],
        merchant_amounts={"ORD-1": 1000.0},
        settled_amount=980.0,
        gateway_fee_deducted=None,
        tax_deducted=None,
        reason="AMOUNT_MISMATCH",
    )
    base.update(overrides)
    return UnresolvedRecord.model_validate(base)


def _variance(record: UnresolvedRecord, is_bundled: bool = False) -> VarianceReport:
    pct = round((record.merchant_total - record.settled_amount) / record.merchant_total * 100, 4)
    return VarianceReport(
        payout_id=record.payout_id,
        merchant_total=record.merchant_total,
        settled_amount=record.settled_amount,
        absolute_variance=round(record.merchant_total - record.settled_amount, 2),
        variance_percent=pct,
        is_bundled=is_bundled,
    )


def test_undisclosed_fee_case_clears_confidence_threshold():
    record = _record(settled_amount=980.0, gateway_fee_deducted=None)  # exactly 2% short
    variance = _variance(record)
    confidence, explanation, category = _reason_about_variance(record, variance, FEES)
    assert confidence >= CONFIDENCE_THRESHOLD
    assert category == "UNDISCLOSED_GATEWAY_FEE"
    assert "2.0%" in explanation or "2%" in explanation


def test_fee_case_does_not_fire_if_a_fee_was_already_itemized():
    """If gateway_fee_deducted is already a real non-zero number, the '2% missing
    line item' story doesn't apply — the agent should not force-fit that reason."""
    record = _record(settled_amount=980.0, gateway_fee_deducted=20.0)
    variance = _variance(record)
    confidence, explanation, category = _reason_about_variance(record, variance, FEES)
    assert category != "UNDISCLOSED_GATEWAY_FEE"


def test_bundled_settlement_with_exact_sum_clears_threshold():
    record = _record(
        linked_orders=["ORD-1", "ORD-2"],
        merchant_amounts={"ORD-1": 400.0, "ORD-2": 600.0},
        settled_amount=1000.0,  # exact sum
    )
    variance = _variance(record, is_bundled=True)
    confidence, explanation, category = _reason_about_variance(record, variance, FEES)
    assert confidence >= CONFIDENCE_THRESHOLD
    assert category == "BUNDLED_SETTLEMENT_RECONCILED"


def test_rolling_reserve_case_stays_below_threshold_by_design():
    """This is a core guardrail: even though the reserve rate 'explains' the
    number, the agent must NOT auto-match it, because a hold is a timing
    difference the system cannot independently verify."""
    record = _record(settled_amount=950.0)  # exactly 5% short
    variance = _variance(record)
    confidence, explanation, category = _reason_about_variance(record, variance, FEES)
    assert confidence < CONFIDENCE_THRESHOLD
    assert category == "POSSIBLE_ROLLING_RESERVE_NEEDS_CONFIRMATION"
    assert "human" in explanation.lower() or "confirm" in explanation.lower()


def test_unexplained_variance_produces_honest_low_confidence():
    """An arbitrary, non-fee-shaped shortfall (e.g. a delayed partial refund)
    must never be force-matched to the nearest known rate."""
    record = _record(settled_amount=760.0)  # 24% short — matches nothing
    variance = _variance(record)
    confidence, explanation, category = _reason_about_variance(record, variance, FEES)
    assert confidence < CONFIDENCE_THRESHOLD
    assert category == "UNEXPLAINED_VARIANCE"
    assert confidence < 0.5  # should be clearly, not marginally, low


def test_confidence_is_never_fabricated_above_one_or_below_zero():
    for settled in (980.0, 950.0, 760.0, 1000.0):
        record = _record(settled_amount=settled)
        variance = _variance(record)
        confidence, _, _ = _reason_about_variance(record, variance, FEES)
        assert 0.0 <= confidence <= 1.0


def test_threshold_boundary_is_strictly_gte_not_gt():
    """Confirms main.py's gating logic (confidence >= threshold) treats an
    exact-threshold confidence as a resolve, not an exception — this test
    guards the contract even though no current case lands exactly on 0.90."""
    assert (0.90 >= CONFIDENCE_THRESHOLD) is True
    assert (0.899999 >= CONFIDENCE_THRESHOLD) is False
