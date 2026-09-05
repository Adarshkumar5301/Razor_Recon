"""
tests/test_mcp_server.py
==========================

Unit tests for the MCP server's tool logic. These call the decorated tool
functions directly (proven callable in isolation — see server.py) rather
than spinning up the full stdio transport, so they run fast and don't
depend on a subprocess. The end-to-end transport itself is already
exercised live by src/main.py and documented in README.md's smoke-test
section.
"""
import pytest

from src.mcp_server import server
from src.models import UnresolvedRecord


@pytest.fixture(autouse=True)
def _seed_unresolved(monkeypatch):
    """Populate the server's in-memory unresolved-record table directly,
    bypassing the on-disk handoff file so these tests are hermetic."""
    records = {
        "PO-FEE": UnresolvedRecord(
            payout_id="PO-FEE",
            linked_orders=["ORD-1"],
            merchant_amounts={"ORD-1": 1000.0},
            settled_amount=980.0,  # exactly 2% short
            gateway_fee_deducted=None,
            tax_deducted=None,
            reason="AMOUNT_MISMATCH",
        ),
        "PO-BUNDLE": UnresolvedRecord(
            payout_id="PO-BUNDLE",
            linked_orders=["ORD-2", "ORD-3"],
            merchant_amounts={"ORD-2": 100.0, "ORD-3": 200.0},
            settled_amount=300.0,  # exact sum, bundled
            gateway_fee_deducted=0.0,
            tax_deducted=0.0,
            reason="BUNDLED_MULTI_ORDER_PAYOUT",
        ),
    }
    monkeypatch.setattr(server, "_UNRESOLVED", records)
    yield


def test_calculate_variance_for_known_payout():
    result = server.calculate_variance("PO-FEE")
    assert result["merchant_total"] == pytest.approx(1000.0)
    assert result["settled_amount"] == pytest.approx(980.0)
    assert result["variance_percent"] == pytest.approx(2.0, abs=0.01)
    assert result["is_bundled"] is False


def test_calculate_variance_flags_bundled_payouts():
    result = server.calculate_variance("PO-BUNDLE")
    assert result["is_bundled"] is True
    assert result["variance_percent"] == pytest.approx(0.0, abs=0.001)


def test_calculate_variance_rejects_unknown_payout_id():
    """This is the security boundary in miniature: the tool must refuse to
    fabricate data for a payout_id it was never given access to."""
    with pytest.raises(ValueError):
        server.calculate_variance("PO-DOES-NOT-EXIST")


def test_fetch_historical_fees_returns_static_schedule():
    schedule = server.fetch_historical_fees()
    assert schedule["standard_gateway_fee_percent"] == 2.0
    assert schedule["rolling_reserve_hold_percent"] == 5.0
    assert schedule["tolerance_percent"] > 0


def test_simulate_tax_deduction_computes_correctly():
    result = server.simulate_tax_deduction(base_amount=100.0, tax_rate_percent=18.0)
    assert result["simulated_tax_amount"] == pytest.approx(18.0)
    assert result["net_of_tax"] == pytest.approx(82.0)


def test_simulate_tax_deduction_rejects_negative_amount():
    with pytest.raises(ValueError):
        server.simulate_tax_deduction(base_amount=-5.0)


def test_list_unresolved_records_only_exposes_seeded_records():
    """The list tool must never expose anything beyond what was loaded —
    proving the MCP boundary doesn't leak the full merchant/payout databases."""
    records = server.list_unresolved_records()
    payout_ids = {r["payout_id"] for r in records}
    assert payout_ids == {"PO-FEE", "PO-BUNDLE"}
