"""
src/agent/features.py
========================

Single source of truth for turning an UnresolvedRecord into the numeric
feature vector the ML classifier consumes. Used identically by:

  - data/training/train_model.py   (training time)
  - src/agent/ml_reasoner.py       (inference time, inside the live agent)

This exists as its own module specifically so training and inference can
never silently drift apart — a classic, easy-to-introduce ML bug where the
model is trained on one feature computation and served with a slightly
different one, producing predictions that look fine in testing and are
subtly wrong in production. Both call sites import `extract_features` from
here and nothing else.
"""
from typing import List

from src.models import UnresolvedRecord

FEATURE_NAMES: List[str] = [
    "variance_percent",
    "abs_variance_percent",
    "is_bundled",
    "num_linked_orders",
    "fee_itemized",
    "settled_ratio",
    "log_merchant_total",
]


def extract_features(record: UnresolvedRecord) -> List[float]:
    """
    Compute the feature vector for one unresolved record. Mirrors exactly
    what the MCP server's `calculate_variance` tool reports to the agent —
    the model only ever sees information the agent itself has legitimate
    access to via the MCP boundary (see AGENTS.md §1.2).
    """
    import math

    merchant_total = record.merchant_total
    settled = record.settled_amount

    variance_percent = 0.0 if merchant_total == 0 else round(
        (merchant_total - settled) / merchant_total * 100, 4
    )
    is_bundled = 1.0 if len(record.linked_orders) > 1 else 0.0
    fee_itemized = 1.0 if (record.gateway_fee_deducted not in (None, 0, 0.0)) else 0.0
    settled_ratio = 0.0 if merchant_total == 0 else round(settled / merchant_total, 6)
    log_merchant_total = math.log1p(max(merchant_total, 0.0))

    return [
        variance_percent,
        abs(variance_percent),
        is_bundled,
        float(len(record.linked_orders)),
        fee_itemized,
        settled_ratio,
        log_merchant_total,
    ]
