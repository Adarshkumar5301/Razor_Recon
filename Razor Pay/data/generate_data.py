"""
data/generate_data.py
======================

Generates the two synthetic datasets required by Phase 2:
  - data/merchant_sales.json   (what the merchant expects to be paid)
  - data/razorpay_payouts.json (what actually landed in the bank account)

This is a SEEDED, deterministic generator (random.seed(42)) — re-running it
produces byte-identical output. That matters for a finance system: a demo
or test run should be reproducible, not a new roll of the dice every time.

Batch composition (50 merchant orders total):
  - 40 orders (80%) settle as a clean, exact 1:1 payout.
    -> These are the records Phase 3's deterministic matcher clears alone.
  - 10 orders (20%) are deliberately messy, split across four archetypes
    that a real Razorpay settlement file actually produces:

      1. UNDISCLOSED_GATEWAY_FEE (4 orders)
         A flat 2% platform fee is deducted with no itemized fee field.
         This is the textbook case the AI agent is supposed to catch by
         cross-referencing the historical fee schedule.

      2. BUNDLED_SETTLEMENT (1 pairing: 2 orders -> 1 payout)
         Two merchant orders are paid out together in a single payout.
         The sum matches exactly, but because it isn't a 1:1 record it
         cannot pass the deterministic layer's strict rule.

      3. ROLLING_RESERVE_HOLD (2 orders)
         A 5% rolling reserve is withheld pending a future release date.
         This looks superficially like the fee case but is NOT a fee —
         the money isn't lost, it's delayed — so the agent should treat
         it with lower confidence and prefer a human review.

      4. DELAYED_PARTIAL_REFUND (2 orders)
         An arbitrary, non-fee-shaped amount is missing because a
         customer refund was processed against the payout but not yet
         reflected in the merchant's own ledger. There is no clean
         percentage that explains this — by design, the agent should
         NOT be able to resolve it with high confidence.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

SEED = 42
TOTAL_ORDERS = 50
CLEAN_COUNT = 40
UNDISCLOSED_FEE_COUNT = 4
BUNDLED_PAIR_COUNT = 1           # -> consumes 2 orders (2 orders per payout)
RESERVE_HOLD_COUNT = 2
DELAYED_REFUND_COUNT = 2
# 40 clean + 4 fee + 2 (1 bundled pair) + 2 reserve + 2 refund = 50 total,
# with exactly 10 "messy" orders -> 20% of the batch, matching the brief.

STANDARD_GATEWAY_FEE_PERCENT = 2.0
ROLLING_RESERVE_PERCENT = 5.0

DATA_DIR = Path(__file__).resolve().parent


def _round2(x: float) -> float:
    return round(x + 1e-9, 2)


def _random_amount(rng: random.Random, low: float = 499.0, high: float = 48999.0) -> float:
    return _round2(rng.uniform(low, high))


def _random_timestamp(rng: random.Random, base: datetime) -> str:
    delta = timedelta(days=rng.randint(0, 29), hours=rng.randint(0, 23), minutes=rng.randint(0, 59))
    return (base + delta).isoformat(timespec="seconds") + "Z"


def generate() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(SEED)
    base_time = datetime(2026, 8, 1, 0, 0, 0)

    merchant_sales: List[Dict[str, Any]] = []
    payouts: List[Dict[str, Any]] = []

    order_counter = 1
    payout_counter = 1

    def next_order_id() -> str:
        nonlocal order_counter
        oid = f"ORD-{order_counter:04d}"
        order_counter += 1
        return oid

    def next_payout_id() -> str:
        nonlocal payout_counter
        pid = f"PAYOUT-{payout_counter:04d}"
        payout_counter += 1
        return pid

    def make_sale(order_id: str) -> Dict[str, Any]:
        return {
            "order_id": order_id,
            "timestamp": _random_timestamp(rng, base_time),
            "amount": _random_amount(rng),
            "customer_id": f"CUST-{rng.randint(1, 30):04d}",
        }

    # 1) Clean, exact 1:1 matches -------------------------------------------------
    for _ in range(CLEAN_COUNT):
        oid = next_order_id()
        sale = make_sale(oid)
        merchant_sales.append(sale)
        payouts.append(
            {
                "payout_id": next_payout_id(),
                "linked_orders": [oid],
                "settled_amount": sale["amount"],
                "gateway_fee_deducted": 0.0,
                "tax_deducted": 0.0,
            }
        )

    # 2) Undisclosed 2% gateway fee -------------------------------------------------
    for _ in range(UNDISCLOSED_FEE_COUNT):
        oid = next_order_id()
        sale = make_sale(oid)
        merchant_sales.append(sale)
        settled = _round2(sale["amount"] * (1 - STANDARD_GATEWAY_FEE_PERCENT / 100))
        payouts.append(
            {
                "payout_id": next_payout_id(),
                "linked_orders": [oid],
                "settled_amount": settled,
                "gateway_fee_deducted": None,  # <- the missing line item
                "tax_deducted": None,
            }
        )

    # 3) Bundled settlements (2 merchant orders -> 1 payout) -----------------------
    for _ in range(BUNDLED_PAIR_COUNT):
        oid_a, oid_b = next_order_id(), next_order_id()
        sale_a, sale_b = make_sale(oid_a), make_sale(oid_b)
        merchant_sales.extend([sale_a, sale_b])
        combined = _round2(sale_a["amount"] + sale_b["amount"])
        payouts.append(
            {
                "payout_id": next_payout_id(),
                "linked_orders": [oid_a, oid_b],
                "settled_amount": combined,
                "gateway_fee_deducted": 0.0,
                "tax_deducted": 0.0,
            }
        )

    # 4) Rolling reserve hold (5% withheld, released later) ------------------------
    for _ in range(RESERVE_HOLD_COUNT):
        oid = next_order_id()
        sale = make_sale(oid)
        merchant_sales.append(sale)
        settled = _round2(sale["amount"] * (1 - ROLLING_RESERVE_PERCENT / 100))
        payouts.append(
            {
                "payout_id": next_payout_id(),
                "linked_orders": [oid],
                "settled_amount": settled,
                "gateway_fee_deducted": 0.0,
                "tax_deducted": 0.0,
            }
        )

    # 5) Delayed partial refund (arbitrary, non-fee-shaped shortfall) --------------
    for _ in range(DELAYED_REFUND_COUNT):
        oid = next_order_id()
        sale = make_sale(oid)
        merchant_sales.append(sale)
        # A refund of a random, unrelated fraction of the order — deliberately NOT
        # shaped like a clean 2% or 5% rate, so no historical schedule explains it.
        refund_fraction = rng.uniform(0.11, 0.34)
        settled = _round2(sale["amount"] * (1 - refund_fraction))
        payouts.append(
            {
                "payout_id": next_payout_id(),
                "linked_orders": [oid],
                "settled_amount": settled,
                "gateway_fee_deducted": 0.0,
                "tax_deducted": 0.0,
            }
        )

    assert len(merchant_sales) == TOTAL_ORDERS, f"expected {TOTAL_ORDERS} orders, got {len(merchant_sales)}"

    # Shuffle payout order only (merchant_sales stays chronological by order_id)
    # so downstream code can't accidentally rely on positional alignment.
    rng.shuffle(payouts)

    return merchant_sales, payouts


def main() -> None:
    merchant_sales, payouts = generate()

    sales_path = DATA_DIR / "merchant_sales.json"
    payouts_path = DATA_DIR / "razorpay_payouts.json"

    sales_path.write_text(json.dumps(merchant_sales, indent=2), encoding="utf-8")
    payouts_path.write_text(json.dumps(payouts, indent=2), encoding="utf-8")

    print(f"Wrote {len(merchant_sales)} merchant sale records -> {sales_path}")
    print(f"Wrote {len(payouts)} payout records -> {payouts_path}")


if __name__ == "__main__":
    main()
