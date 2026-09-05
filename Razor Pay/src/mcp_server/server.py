"""
src/mcp_server/server.py
==========================

Phase 4 — The Model Context Protocol (MCP) server.

This is the security checkpoint described in the brief: the "USB-C port"
that lets the AI agent investigate ambiguous settlement records *in place*,
without ever touching merchant_sales.json or razorpay_payouts.json
directly, and without any financial data leaving this local process.

Boundary contract
------------------
- The agent can list unresolved records (`list_unresolved_records`) and
  call three narrow, read-only analysis tools against them. It cannot
  read arbitrary files, it cannot see the 40 already-matched clean
  records, and it cannot write anything back through this server —
  writing the final ledger is main.py's job, not the agent's.
- Every tool call is validated (Pydantic) and logged to stderr with a
  timestamp, tool name, and arguments, so every AI decision is traceable
  back to the exact evidence it was given. This log is the seed of the
  audit trail in Phase 6.
- `fetch_historical_fees` and `simulate_tax_deduction` are pure, local,
  offline computations against a hard-coded rate table. Nothing here ever
  makes an outbound network call — there is no hidden dependency on a
  live Razorpay API, by design (see AGENTS.md, "No network calls from
  the MCP server").

Run standalone for a manual smoke test:
    python -m src.mcp_server.server
(it will idle on stdio waiting for a client — Ctrl+C to exit)
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from mcp.server.fastmcp import FastMCP

from src.models import (
    HistoricalFeeSchedule,
    TaxSimulationResult,
    UnresolvedRecord,
    VarianceReport,
)

# ---------------------------------------------------------------------------
# Local, offline, hard-coded knowledge base.
# This models Razorpay's publicly documented standard rate card. It is a
# static lookup table, not a live API call — see module docstring.
# ---------------------------------------------------------------------------
_FEE_SCHEDULE = HistoricalFeeSchedule(
    standard_gateway_fee_percent=2.0,
    gst_on_fee_percent=18.0,
    rolling_reserve_hold_percent=5.0,
    rolling_reserve_release_days=180,
    tolerance_percent=0.35,
)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _log(event: str, **fields: object) -> None:
    """Structured, timestamped audit log line. Written to stderr so it never
    contaminates the MCP stdio protocol channel on stdout."""
    payload = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    print(f"[mcp_server] {json.dumps(payload, default=str)}", file=sys.stderr, flush=True)


def _load_unresolved_records() -> Dict[str, UnresolvedRecord]:
    """Load the unresolved records produced by Phase 3, keyed by payout_id.

    This is intentionally the ONLY data this server reads from disk. It never
    opens merchant_sales.json or razorpay_payouts.json directly — those raw
    files are outside this server's exposed surface area entirely.
    """
    path = _DATA_DIR / "deterministic_result.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.deterministic.matcher` first "
            "to produce the unresolved-records handoff file."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = [UnresolvedRecord.model_validate(r) for r in raw["unresolved"]]
    return {r.payout_id: r for r in records}


# Loaded once at server startup; this process is short-lived (one reconciliation
# run) so there is no staleness concern in re-reading it per request.
_UNRESOLVED: Dict[str, UnresolvedRecord] = {}

mcp = FastMCP(
    name="razorpay-reconciliation-mcp",
    instructions=(
        "You have access to exactly the unresolved settlement records for this "
        "reconciliation batch. Use calculate_variance, fetch_historical_fees, and "
        "simulate_tax_deduction to investigate a record before proposing a match. "
        "You cannot access any other financial data through this server."
    ),
)


@mcp.tool()
def list_unresolved_records() -> List[dict]:
    """
    Return every unresolved settlement record this server is permitted to expose.

    This is the ONLY way the agent can see payout data — there is no tool to
    read arbitrary files or query the full merchant/payout databases.
    """
    _log("list_unresolved_records", count=len(_UNRESOLVED))
    return [r.model_dump() for r in _UNRESOLVED.values()]


@mcp.tool()
def calculate_variance(payout_id: str) -> dict:
    """
    Compute the exact monetary and percentage variance between what the
    merchant expected (sum of all linked orders) and what Razorpay actually
    settled, for a single unresolved payout.

    Args:
        payout_id: The payout to analyze (must be one of the ids returned by
            list_unresolved_records).
    """
    record = _UNRESOLVED.get(payout_id)
    if record is None:
        _log("calculate_variance.error", payout_id=payout_id, error="not_found")
        raise ValueError(f"'{payout_id}' is not a known unresolved record.")

    merchant_total = record.merchant_total
    variance_percent = 0.0 if merchant_total == 0 else round(
        (merchant_total - record.settled_amount) / merchant_total * 100, 4
    )
    report = VarianceReport(
        payout_id=payout_id,
        merchant_total=merchant_total,
        settled_amount=record.settled_amount,
        absolute_variance=record.raw_variance,
        variance_percent=variance_percent,
        is_bundled=len(record.linked_orders) > 1,
    )
    _log("calculate_variance", payout_id=payout_id, variance_percent=variance_percent)
    return report.model_dump()


@mcp.tool()
def fetch_historical_fees() -> dict:
    """
    Return Razorpay's known standard rate card: the typical gateway fee
    percentage, GST rate applied on top of fees, and the rolling reserve
    hold percentage + release window. This is local, offline reference
    data the agent can compare an observed variance against.
    """
    _log("fetch_historical_fees")
    return _FEE_SCHEDULE.model_dump()


@mcp.tool()
def simulate_tax_deduction(base_amount: float, tax_rate_percent: float = 18.0) -> dict:
    """
    Compute what a tax deduction (e.g. GST charged on a gateway fee) of a
    given rate would look like against a given base amount, so the agent
    can check whether a variance is explained by "fee + tax on that fee"
    rather than fee alone.

    Args:
        base_amount: The amount the tax would be calculated on (e.g. the
            gateway fee itself, not the full order amount).
        tax_rate_percent: The tax rate to apply. Defaults to 18% (standard
            Indian GST rate on financial services).
    """
    if base_amount < 0:
        raise ValueError("base_amount must be non-negative.")
    if tax_rate_percent < 0:
        raise ValueError("tax_rate_percent must be non-negative.")

    simulated_tax = round(base_amount * tax_rate_percent / 100, 2)
    result = TaxSimulationResult(
        base_amount=base_amount,
        tax_rate_percent=tax_rate_percent,
        simulated_tax_amount=simulated_tax,
        net_of_tax=round(base_amount - simulated_tax, 2),
    )
    _log("simulate_tax_deduction", base_amount=base_amount, tax_rate_percent=tax_rate_percent)
    return result.model_dump()


def _bootstrap() -> None:
    global _UNRESOLVED
    _UNRESOLVED = _load_unresolved_records()
    _log("server_bootstrap", unresolved_count=len(_UNRESOLVED))


if __name__ == "__main__":
    _bootstrap()
    mcp.run(transport="stdio")
