"""
src/deterministic/matcher.py
=============================

Phase 3 — The Deterministic Matcher (Base Layer).

This module contains ZERO AI, ZERO heuristics, and ZERO tolerance bands.
It exists purely to clear the records a human auditor would clear on sight:
exactly one payout, linked to exactly one order, for exactly the amount the
merchant expected. Everything else — by design, not by accident — is
pushed downstream to `unresolved_records` for the AI agent in Phase 5 to
investigate via the MCP server.

Money is compared using `Decimal`, never raw `float`, to avoid binary
floating-point rounding artifacts producing false mismatches on numbers
that are financially identical (e.g. 100.10 vs 100.099999999999).
"""
from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List

from src.models import (
    DeterministicMatchResult,
    MatchedRecord,
    MerchantSale,
    RazorpayPayout,
    UnresolvedRecord,
)


def _to_money(value: float) -> Decimal:
    """Convert a float amount to an exact 2-decimal-place Decimal for safe comparison."""
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def load_merchant_sales(path: Path) -> Dict[str, MerchantSale]:
    """Load merchant_sales.json into an order_id-keyed lookup table."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    sales = [MerchantSale.model_validate(row) for row in raw]
    order_ids = [sale.order_id for sale in sales]
    if len(order_ids) != len(set(order_ids)):
        raise ValueError("merchant_sales.json contains duplicate order_id values.")
    return {sale.order_id: sale for sale in sales}


def load_payouts(path: Path) -> List[RazorpayPayout]:
    """Load razorpay_payouts.json into a list of validated payout records."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    payouts = [RazorpayPayout.model_validate(row) for row in raw]
    payout_ids = [payout.payout_id for payout in payouts]
    if len(payout_ids) != len(set(payout_ids)):
        raise ValueError("razorpay_payouts.json contains duplicate payout_id values.")
    return payouts


def _build_unresolved(payout: RazorpayPayout, sales: Dict[str, MerchantSale], reason: str) -> UnresolvedRecord:
    """Assemble an UnresolvedRecord, pulling in whatever merchant amounts we DO
    have for the payout's linked orders (some may legitimately be missing)."""
    merchant_amounts = {oid: sales[oid].amount for oid in payout.linked_orders if oid in sales}
    return UnresolvedRecord(
        payout_id=payout.payout_id,
        linked_orders=payout.linked_orders,
        merchant_amounts=merchant_amounts,
        settled_amount=payout.settled_amount,
        gateway_fee_deducted=payout.gateway_fee_deducted,
        tax_deducted=payout.tax_deducted,
        reason=reason,
    )


def run_deterministic_match(merchant_sales_path: Path, razorpay_payouts_path: Path) -> DeterministicMatchResult:
    """
    Execute the deterministic matching pass.

    A payout is CLEARED (added to `matched`) if and only if:
      1. It links to exactly one order, AND
      2. That order exists in the merchant ledger, AND
      3. settled_amount == merchant amount, to the cent, with zero tolerance.

    Every other payout — bundled, mismatched, or referencing an unknown
    order — is routed to `unresolved`. Merchant orders that never appear in
    ANY payout at all are also surfaced as unresolved (money the merchant
    expects that Razorpay's settlement file never mentions).
    """
    sales = load_merchant_sales(merchant_sales_path)
    payouts = load_payouts(razorpay_payouts_path)

    matched: List[MatchedRecord] = []
    unresolved: List[UnresolvedRecord] = []
    orders_seen_in_a_payout: set[str] = set()
    order_reference_count = {
        order_id: sum(order_id in payout.linked_orders for payout in payouts)
        for payout in payouts
        for order_id in payout.linked_orders
    }

    for payout in payouts:
        orders_seen_in_a_payout.update(payout.linked_orders)

        # A sale linked to more than one payout is ambiguous.  Never clear
        # either record automatically: that could double-count revenue.
        if any(order_reference_count[order_id] > 1 for order_id in payout.linked_orders):
            unresolved.append(
                _build_unresolved(payout, sales, reason="ORDER_APPEARS_IN_MULTIPLE_PAYOUTS")
            )
            continue

        if len(payout.linked_orders) != 1:
            unresolved.append(_build_unresolved(payout, sales, reason="BUNDLED_MULTI_ORDER_PAYOUT"))
            continue

        order_id = payout.linked_orders[0]
        sale = sales.get(order_id)
        if sale is None:
            unresolved.append(_build_unresolved(payout, sales, reason="ORDER_NOT_FOUND_IN_MERCHANT_LEDGER"))
            continue

        if _to_money(sale.amount) == _to_money(payout.settled_amount):
            matched.append(
                MatchedRecord(
                    order_ids=[order_id],
                    payout_id=payout.payout_id,
                    merchant_total=sale.amount,
                    settled_amount=payout.settled_amount,
                )
            )
        else:
            unresolved.append(_build_unresolved(payout, sales, reason="AMOUNT_MISMATCH"))

    # Orders the merchant recorded that never show up in ANY payout at all.
    for order_id, sale in sales.items():
        if order_id not in orders_seen_in_a_payout:
            unresolved.append(
                UnresolvedRecord(
                    # Keep this identifier unique: the MCP server indexes its
                    # restricted records by payout_id, and multiple missing
                    # orders must never overwrite one another in that boundary.
                    payout_id=f"NO_PAYOUT_FOUND:{order_id}",
                    linked_orders=[order_id],
                    merchant_amounts={order_id: sale.amount},
                    settled_amount=0.0,
                    gateway_fee_deducted=None,
                    tax_deducted=None,
                    reason="NO_PAYOUT_FOUND_FOR_ORDER",
                )
            )

    return DeterministicMatchResult(
        matched=matched,
        unresolved=unresolved,
        total_merchant_orders=len(sales),
        total_payouts=len(payouts),
    )


def _default_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "merchant_sales.json", root / "data" / "razorpay_payouts.json"


if __name__ == "__main__":
    sales_path, payouts_path = _default_paths()
    result = run_deterministic_match(sales_path, payouts_path)

    out_path = sales_path.parent / "deterministic_result.json"
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    print(f"Matched (deterministic): {len(result.matched)}")
    print(f"Unresolved (-> AI layer): {len(result.unresolved)}")
    print(f"Deterministic match rate: {len(result.matched) / result.total_payouts:.1%}")
    print(f"Wrote {out_path}")
