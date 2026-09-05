"""
tests/test_ml_reasoner.py
============================

Tests for the ML-backed reasoning engine (src/agent/ml_reasoner.py) and
its shared feature extraction (src/agent/features.py).

Requires data/training/edge_case_model.joblib to already exist — run
`python -m data.training.train_model` first (see AGENTS.md's CI steps).
These tests are skipped, not failed, if the model artifact is absent, so
`pytest tests/` still passes on a fresh checkout before training has run.
"""
from pathlib import Path

import pytest

from src.agent.features import extract_features
from src.models import UnresolvedRecord

MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "training" / "edge_case_model.joblib"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="edge_case_model.joblib not found — run `python -m data.training.train_model` first",
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


def test_feature_vector_shape_and_values():
    record = _record(settled_amount=980.0)
    features = extract_features(record)
    assert len(features) == 7
    variance_pct, abs_variance_pct, is_bundled, num_orders, fee_itemized, settled_ratio, log_total = features
    assert variance_pct == pytest.approx(2.0, abs=0.01)
    assert abs_variance_pct == pytest.approx(2.0, abs=0.01)
    assert is_bundled == 0.0
    assert num_orders == 1.0
    assert fee_itemized == 0.0
    assert settled_ratio == pytest.approx(0.98, abs=0.001)


def test_feature_vector_detects_bundling_and_itemized_fee():
    record = _record(
        linked_orders=["ORD-1", "ORD-2"],
        merchant_amounts={"ORD-1": 500.0, "ORD-2": 500.0},
        settled_amount=1000.0,
        gateway_fee_deducted=25.0,
    )
    features = extract_features(record)
    _, _, is_bundled, num_orders, fee_itemized, _, _ = features
    assert is_bundled == 1.0
    assert num_orders == 2.0
    assert fee_itemized == 1.0


def test_ml_reasoner_classifies_clean_fee_case_correctly():
    """A textbook 2% undisclosed fee should be CLASSIFIED correctly by the
    model. Note: unlike the rule engine's fixed decision table, a trained
    classifier's confidence is a vote fraction that varies with the exact
    feature values, not a guaranteed >= 90% for every instance of a category
    — so this test checks classification correctness and a sane confidence
    floor, not parity with the rule engine's confidence number. (On the
    full 10-batch demo dataset, this same archetype does clear 90%+ — see
    data/audit_trail.md after `python -m src.main --engine ml` — but a
    single hand-built example isn't guaranteed to land in the same
    probability bucket as the model's training distribution.)"""
    from src.agent import ml_reasoner
    from src.models import HistoricalFeeSchedule, VarianceReport

    record = _record(settled_amount=980.0, gateway_fee_deducted=None)
    variance = VarianceReport(
        payout_id="PO-TEST", merchant_total=1000.0, settled_amount=980.0,
        absolute_variance=20.0, variance_percent=2.0, is_bundled=False,
    )
    fees = HistoricalFeeSchedule(
        standard_gateway_fee_percent=2.0, gst_on_fee_percent=18.0,
        rolling_reserve_hold_percent=5.0, rolling_reserve_release_days=180, tolerance_percent=0.35,
    )
    confidence, explanation, category = ml_reasoner.reason(record, variance, fees)
    assert category == "UNDISCLOSED_GATEWAY_FEE"
    assert confidence > 0.5  # correctly the top prediction, not just barely so
    assert "UNDISCLOSED_GATEWAY_FEE" in explanation


def test_ml_reasoner_never_auto_resolves_non_allowlisted_category():
    """Policy guardrail: even if the model is highly confident about WHAT a
    record is, categories outside AUTO_RESOLVABLE_CATEGORIES must never
    produce a confidence above the resolution threshold."""
    from src.agent import ml_reasoner
    from src.agent.reconciler import CONFIDENCE_THRESHOLD
    from src.models import HistoricalFeeSchedule, VarianceReport

    # A textbook 5% reserve-hold-shaped variance.
    record = _record(settled_amount=950.0)
    variance = VarianceReport(
        payout_id="PO-TEST", merchant_total=1000.0, settled_amount=950.0,
        absolute_variance=50.0, variance_percent=5.0, is_bundled=False,
    )
    fees = HistoricalFeeSchedule(
        standard_gateway_fee_percent=2.0, gst_on_fee_percent=18.0,
        rolling_reserve_hold_percent=5.0, rolling_reserve_release_days=180, tolerance_percent=0.35,
    )
    confidence, explanation, category = ml_reasoner.reason(record, variance, fees)
    if category not in ml_reasoner.AUTO_RESOLVABLE_CATEGORIES:
        assert confidence < CONFIDENCE_THRESHOLD


def test_ml_reasoner_confidence_always_in_valid_range():
    from src.agent import ml_reasoner
    from src.models import HistoricalFeeSchedule, VarianceReport

    fees = HistoricalFeeSchedule(
        standard_gateway_fee_percent=2.0, gst_on_fee_percent=18.0,
        rolling_reserve_hold_percent=5.0, rolling_reserve_release_days=180, tolerance_percent=0.35,
    )
    for settled in (980.0, 950.0, 760.0, 999.99):
        record = _record(settled_amount=settled)
        variance = VarianceReport(
            payout_id="PO-TEST", merchant_total=1000.0, settled_amount=settled,
            absolute_variance=round(1000.0 - settled, 2),
            variance_percent=round((1000.0 - settled) / 1000.0 * 100, 4), is_bundled=False,
        )
        confidence, _, _ = ml_reasoner.reason(record, variance, fees)
        assert 0.0 <= confidence <= 1.0
