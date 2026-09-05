"""
data/training/generate_training_batches.py
=============================================

Generates 10 dummy datasets ("batches") for training the AI agent's
reasoning model on edge cases.

This is deliberately DIFFERENT from data/generate_data.py, which produces
one clean demo batch with a fixed 2.0% fee / 5.0% reserve rate. Here, every
batch draws its fee rate, reserve rate, and refund fraction from WIDE,
randomized ranges (and per-record jitter on top of that), so the resulting
50-record-per-batch x 10-batch corpus actually contains edge cases: fees at
1.1%, fees at 3.4%, reserve holds at 3.2%, at 6.8%, etc. A model trained
only on exact 2.0%/5.0% examples would just be a lookup table wearing a
sklearn costume — this generator exists so the model has to learn a real
decision boundary instead.

Two new archetypes are added beyond the four in the original demo dataset,
specifically because they're common real-world edge cases that are easy to
confuse with the others:

  5. ROUNDING_DRIFT
     A sub-paisa-scale variance (0.00%-0.05%) caused by floating point /
     currency-conversion rounding, NOT a fee or a hold. Looks superficially
     like "yet another small mismatch" but should be auto-clearable with
     very high confidence — the deterministic layer misses it only because
     it demands byte-exact equality.

  6. FEE_PLUS_RESERVE_COMBINED
     Both an undisclosed gateway fee AND a rolling reserve hold are applied
     to the same payout. This is a genuinely hard edge case: the total
     variance is a fee-shaped percentage PLUS a reserve-shaped percentage,
     which doesn't cleanly match either known single rate. A rule-based
     system with fixed tolerance bands is likely to file this under
     UNEXPLAINED_VARIANCE; whether a learned model can do better is exactly
     the kind of thing worth measuring with a held-out test set.

Ground truth labels are written to a sidecar `labels.json` per batch
(payout_id -> archetype), NEVER into merchant_sales.json / razorpay_
payouts.json themselves — those two files must stay in the exact same
production schema (src/models.py) that src/deterministic/matcher.py
consumes, so training data generation can reuse the real matcher instead
of a parallel reimplementation.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

NUM_BATCHES = 10
ORDERS_PER_BATCH = 50
BASE_SEED = 100  # batches use seeds 100..109 -> fully reproducible, distinct from the seed=42 demo set

TRAINING_DIR = Path(__file__).resolve().parent


def _round2(x: float) -> float:
    return round(x + 1e-9, 2)


def _random_amount(rng: random.Random, low: float = 300.0, high: float = 60000.0) -> float:
    return _round2(rng.uniform(low, high))


def _random_timestamp(rng: random.Random, base: datetime) -> str:
    delta = timedelta(days=rng.randint(0, 59), hours=rng.randint(0, 23), minutes=rng.randint(0, 59))
    return (base + delta).isoformat(timespec="seconds") + "Z"


def _generate_batch(batch_index: int) -> Tuple[List[dict], List[dict], Dict[str, str]]:
    """Generate one batch. Returns (merchant_sales, payouts, labels)."""
    seed = BASE_SEED + batch_index
    rng = random.Random(seed)
    base_time = datetime(2026, 1, 1) + timedelta(days=30 * batch_index)

    # Per-batch archetype mix is itself randomized (within sane bounds) so the
    # 10-batch corpus varies in composition, not just in individual amounts.
    clean_count = rng.randint(30, 36)
    fee_count = rng.randint(4, 7)
    bundle_pair_count = rng.randint(1, 2)
    reserve_count = rng.randint(2, 4)
    refund_count = rng.randint(2, 4)
    rounding_count = rng.randint(1, 3)
    combined_count = rng.randint(1, 2)

    merchant_sales: List[dict] = []
    payouts: List[dict] = []
    labels: Dict[str, str] = {}

    order_counter = 1
    payout_counter = 1

    def next_order_id() -> str:
        nonlocal order_counter
        oid = f"B{batch_index:02d}-ORD-{order_counter:04d}"
        order_counter += 1
        return oid

    def next_payout_id() -> str:
        nonlocal payout_counter
        pid = f"B{batch_index:02d}-PAYOUT-{payout_counter:04d}"
        payout_counter += 1
        return pid

    def make_sale(order_id: str) -> dict:
        return {
            "order_id": order_id,
            "timestamp": _random_timestamp(rng, base_time),
            "amount": _random_amount(rng),
            "customer_id": f"B{batch_index:02d}-CUST-{rng.randint(1, 40):04d}",
        }

    # 1) Clean exact matches ------------------------------------------------
    for _ in range(clean_count):
        oid = next_order_id()
        sale = make_sale(oid)
        merchant_sales.append(sale)
        payouts.append({
            "payout_id": next_payout_id(), "linked_orders": [oid],
            "settled_amount": sale["amount"], "gateway_fee_deducted": 0.0, "tax_deducted": 0.0,
        })
        # No label needed: the deterministic matcher clears these before the
        # AI/ML layer ever sees them.

    # 2) Undisclosed gateway fee, WIDE randomized rate (1.0%-3.5%) ----------
    for _ in range(fee_count):
        oid = next_order_id()
        sale = make_sale(oid)
        merchant_sales.append(sale)
        fee_pct = rng.uniform(1.0, 3.5)
        settled = _round2(sale["amount"] * (1 - fee_pct / 100))
        pid = next_payout_id()
        payouts.append({
            "payout_id": pid, "linked_orders": [oid],
            "settled_amount": settled, "gateway_fee_deducted": None, "tax_deducted": None,
        })
        labels[pid] = "UNDISCLOSED_GATEWAY_FEE"

    # 3) Bundled settlements (2 orders -> 1 payout), exact sum ---------------
    for _ in range(bundle_pair_count):
        oid_a, oid_b = next_order_id(), next_order_id()
        sale_a, sale_b = make_sale(oid_a), make_sale(oid_b)
        merchant_sales.extend([sale_a, sale_b])
        combined = _round2(sale_a["amount"] + sale_b["amount"])
        pid = next_payout_id()
        payouts.append({
            "payout_id": pid, "linked_orders": [oid_a, oid_b],
            "settled_amount": combined, "gateway_fee_deducted": 0.0, "tax_deducted": 0.0,
        })
        labels[pid] = "BUNDLED_SETTLEMENT_RECONCILED"

    # 4) Rolling reserve hold, WIDE randomized rate (3.0%-7.0%) --------------
    for _ in range(reserve_count):
        oid = next_order_id()
        sale = make_sale(oid)
        merchant_sales.append(sale)
        reserve_pct = rng.uniform(3.0, 7.0)
        settled = _round2(sale["amount"] * (1 - reserve_pct / 100))
        pid = next_payout_id()
        payouts.append({
            "payout_id": pid, "linked_orders": [oid],
            "settled_amount": settled, "gateway_fee_deducted": 0.0, "tax_deducted": 0.0,
        })
        labels[pid] = "ROLLING_RESERVE_HOLD"

    # 5) Delayed/partial refund, arbitrary non-fee-shaped shortfall ---------
    for _ in range(refund_count):
        oid = next_order_id()
        sale = make_sale(oid)
        merchant_sales.append(sale)
        refund_fraction = rng.uniform(0.08, 0.40)
        settled = _round2(sale["amount"] * (1 - refund_fraction))
        pid = next_payout_id()
        payouts.append({
            "payout_id": pid, "linked_orders": [oid],
            "settled_amount": settled, "gateway_fee_deducted": 0.0, "tax_deducted": 0.0,
        })
        labels[pid] = "UNEXPLAINED_VARIANCE"

    # 6) Rounding drift: sub-paisa-scale variance (0.00%-0.05%) -------------
    for _ in range(rounding_count):
        oid = next_order_id()
        sale = make_sale(oid)
        merchant_sales.append(sale)
        drift_pct = rng.uniform(0.001, 0.05)
        settled = _round2(sale["amount"] * (1 - drift_pct / 100))
        if settled == sale["amount"]:
            # Amount rounded away to nothing at 2dp -> force 1 paisa of drift
            # so this row still lands in "unresolved" and stays a useful example.
            settled = _round2(sale["amount"] - 0.01)
        pid = next_payout_id()
        payouts.append({
            "payout_id": pid, "linked_orders": [oid],
            "settled_amount": settled, "gateway_fee_deducted": 0.0, "tax_deducted": 0.0,
        })
        labels[pid] = "ROUNDING_DRIFT"

    # 7) Hard edge case: fee AND reserve hold combined on the same payout ---
    for _ in range(combined_count):
        oid = next_order_id()
        sale = make_sale(oid)
        merchant_sales.append(sale)
        fee_pct = rng.uniform(1.0, 3.5)
        reserve_pct = rng.uniform(3.0, 7.0)
        settled = _round2(sale["amount"] * (1 - (fee_pct + reserve_pct) / 100))
        pid = next_payout_id()
        payouts.append({
            "payout_id": pid, "linked_orders": [oid],
            "settled_amount": settled, "gateway_fee_deducted": None, "tax_deducted": None,
        })
        labels[pid] = "FEE_PLUS_RESERVE_COMBINED"

    rng.shuffle(payouts)
    return merchant_sales, payouts, labels


def generate_all_batches() -> None:
    manifest = []
    for i in range(NUM_BATCHES):
        merchant_sales, payouts, labels = _generate_batch(i)
        batch_dir = TRAINING_DIR / f"batch_{i:02d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        (batch_dir / "merchant_sales.json").write_text(json.dumps(merchant_sales, indent=2), encoding="utf-8")
        (batch_dir / "razorpay_payouts.json").write_text(json.dumps(payouts, indent=2), encoding="utf-8")
        (batch_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

        manifest.append({
            "batch": i, "seed": BASE_SEED + i,
            "orders": len(merchant_sales), "payouts": len(payouts), "labeled_edge_cases": len(labels),
        })
        print(f"batch_{i:02d}: {len(merchant_sales)} orders, {len(payouts)} payouts, "
              f"{len(labels)} labeled edge cases")

    (TRAINING_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total_edge_cases = sum(m["labeled_edge_cases"] for m in manifest)
    print(f"\nWrote {NUM_BATCHES} batches -> {TRAINING_DIR}")
    print(f"Total labeled edge-case records across all batches: {total_edge_cases}")


if __name__ == "__main__":
    generate_all_batches()
