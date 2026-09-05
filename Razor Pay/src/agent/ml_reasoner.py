"""
src/agent/ml_reasoner.py
===========================

A trained-model alternative to `_reason_about_variance` in
src/agent/reconciler.py. Loads the classifier produced by
data/training/train_model.py and serves it at inference time, using the
exact same feature extraction code (src/agent/features.py) the training
script used — no train/serve skew.

Same interface as the rule-based reasoner on purpose: `reason(record,
variance, fees) -> (confidence, explanation, category)`. src/main.py picks
which engine to use via `--engine {rules,ml}`; nothing else in the pipeline
needs to know which one is active.

Guardrail: model confidence is NOT the whole story. Some predicted
categories represent a TIMING difference (a reserve hold) or a genuinely
unexplained gap (unrecorded refund) that this system has no way to verify
independently, no matter how confident the classifier is that it has
correctly IDENTIFIED the category. So there are two questions, not one:
  1. What category does the model think this is? (classification)
  2. Is that category even eligible for auto-resolution? (business policy)
A high-confidence "this looks like a reserve hold" is still routed to the
Honest Exception List, by design — see AUTO_RESOLVABLE_CATEGORIES below.
This mirrors the same guardrail philosophy as the rule-based engine, just
applied after a learned classification instead of a hand-coded one.
"""
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np

from src.agent.features import extract_features
from src.models import HistoricalFeeSchedule, UnresolvedRecord, VarianceReport

MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "training" / "edge_case_model.joblib"

# Categories the model can predict that are safe to auto-resolve even at
# high confidence. Everything else — however confidently classified — is
# a timing difference or genuinely unexplained gap that stays a human call.
AUTO_RESOLVABLE_CATEGORIES = {
    "UNDISCLOSED_GATEWAY_FEE",
    "BUNDLED_SETTLEMENT_RECONCILED",
    "ROUNDING_DRIFT",
}

_EXPLANATION_TEMPLATES = {
    "UNDISCLOSED_GATEWAY_FEE": (
        "ML classifier (trained on {n_train} labeled edge cases across 10 dummy batches) "
        "predicted UNDISCLOSED_GATEWAY_FEE with {conf:.0%} confidence, based on a variance of "
        "{variance_pct:.3f}% and no itemized fee on this payout. Reconciled as: settled_amount = "
        "merchant_amount - gateway fee."
    ),
    "BUNDLED_SETTLEMENT_RECONCILED": (
        "ML classifier predicted BUNDLED_SETTLEMENT_RECONCILED with {conf:.0%} confidence: "
        "{n_orders} linked orders, settled amount reconciles with their combined total "
        "(variance {variance_pct:.3f}%)."
    ),
    "ROUNDING_DRIFT": (
        "ML classifier predicted ROUNDING_DRIFT with {conf:.0%} confidence: the {variance_pct:.4f}% "
        "variance is sub-paisa-scale and consistent with rounding rather than a fee or hold. "
        "Safe to auto-clear."
    ),
    "ROLLING_RESERVE_HOLD": (
        "ML classifier predicted ROLLING_RESERVE_HOLD with {conf:.0%} confidence "
        "(variance {variance_pct:.3f}%). Per policy, reserve-hold predictions are NEVER "
        "auto-resolved regardless of model confidence — a hold is a timing difference this "
        "system cannot independently confirm the release date for. Routed to human review."
    ),
    "UNEXPLAINED_VARIANCE": (
        "ML classifier predicted UNEXPLAINED_VARIANCE with {conf:.0%} confidence "
        "(variance {variance_pct:.3f}%) — no known fee/reserve/bundle pattern explains this "
        "shortfall. Routed to human review rather than guessing."
    ),
    "FEE_PLUS_RESERVE_COMBINED": (
        "ML classifier predicted FEE_PLUS_RESERVE_COMBINED with {conf:.0%} confidence "
        "(variance {variance_pct:.3f}%) — this looks like BOTH an undisclosed fee and a reserve "
        "hold stacked on one payout. This is the hardest edge case in the training set (held-out "
        "test F1 ≈ 0.5) and is never auto-resolved regardless of confidence — routed to human "
        "review for manual decomposition."
    ),
}

_model_bundle: Optional[dict] = None


def _load_model() -> dict:
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} not found. Run `python -m data.training.train_model` first "
                "to train and save the edge-case classifier."
            )
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


def reason(
    record: UnresolvedRecord,
    variance: VarianceReport,
    fees: HistoricalFeeSchedule,  # noqa: ARG001 — kept for interface parity with the rule-based reasoner
) -> Tuple[float, str, str]:
    """
    ML-backed equivalent of `_reason_about_variance`. Same return contract:
    (confidence, human-readable explanation, category).
    """
    bundle = _load_model()
    model = bundle["model"]
    encoder = bundle["encoder"]

    features = np.array([extract_features(record)])
    proba = model.predict_proba(features)[0]
    predicted_idx = int(np.argmax(proba))
    predicted_category = str(encoder.classes_[predicted_idx])
    model_confidence = float(proba[predicted_idx])

    # Business policy gate: cap confidence to a non-auto-resolving level for
    # any category outside the allow-list, no matter what the model says.
    if predicted_category not in AUTO_RESOLVABLE_CATEGORIES:
        effective_confidence = min(model_confidence, 0.65)
    else:
        effective_confidence = model_confidence

    template = _EXPLANATION_TEMPLATES.get(
        predicted_category,
        "ML classifier predicted {predicted_category} with {conf:.0%} confidence.",
    )
    explanation = template.format(
        n_train=139,  # from the committed training report; see data/training/model_report.md
        conf=model_confidence,
        variance_pct=variance.variance_percent,
        n_orders=len(record.linked_orders),
        predicted_category=predicted_category,
    )
    if predicted_category not in AUTO_RESOLVABLE_CATEGORIES and effective_confidence != model_confidence:
        explanation += (
            f" (Model's raw classification confidence was {model_confidence:.0%}; capped to "
            f"{effective_confidence:.0%} by policy since this category is never auto-resolved.)"
        )

    return effective_confidence, explanation, predicted_category
