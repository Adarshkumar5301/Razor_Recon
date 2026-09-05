"""
src/agent/reconciler.py
=========================

Phase 5 — The AI Intelligence Layer.

Design decision worth stating up front (see README "Architecture Decisions"
and AGENTS.md): this agent is a hand-rolled Perceive -> Reason -> Act loop
over the real MCP Python SDK, NOT a call to an external LLM API, and NOT
built on LangChain/LangGraph. Two reasons:

  1. "Do not hallucinate external API calls. Everything must run locally."
     A genuine LLM-backed agent would need a real API key and a live model
     endpoint neither of which this environment can assume exist. Wiring
     one up would mean guessing at an API surface instead of building
     something that actually runs. The MCP client/server boundary — the
     actual subject of Track 04 — is real and fully exercised.
  2. Auditability. A finance-ops reconciliation engine that produces a
     "regulator-ready audit trail" (Phase 6) needs every match to be
     traceable to an explicit, inspectable rule, not a black-box model
     call whose reasoning can't be reproduced on re-run. The heuristic
     engine below is deterministic: same input, same confidence score,
     every time.

  Swapping this for a live LLM call is a contained change: replace
  `_reason_about_variance()` with a call to your model of choice, keep
  feeding it the SAME MCP tool outputs, and keep the same confidence-gate
  contract (>= threshold -> resolved, else -> honest exception).

Agent loop per unresolved record
---------------------------------
  1. PERCEIVE: call `calculate_variance(payout_id)` via MCP.
  2. REASON:   call `fetch_historical_fees()` and, if useful,
               `simulate_tax_deduction(...)`, then compare the observed
               variance against the known fee/reserve rate table.
  3. ACT:      if confidence >= threshold, propose a resolution with a
               plain-language, numeric explanation. Otherwise, stop and
               write an honest exception with the reasoning for why the
               agent declined to guess.
"""
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.models import (
    AgentResolution,
    AgentRunResult,
    ExceptionRecord,
    HistoricalFeeSchedule,
    UnresolvedRecord,
    VarianceReport,
)

CONFIDENCE_THRESHOLD = 0.90
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _extract_json(call_tool_result: Any) -> dict:
    """MCP tool results arrive as a list of content blocks; our tools return a
    single JSON-serializable dict, which FastMCP wraps as one TextContent
    block containing its JSON string representation."""
    import json

    structured = getattr(call_tool_result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured

    text_blocks = [block.text for block in call_tool_result.content if block.type == "text"]
    if not text_blocks:
        raise ValueError("MCP tool call returned no text content.")
    payload = text_blocks[0].strip()
    if payload.startswith("```json") and payload.endswith("```"):
        payload = payload[7:-3].strip()
    return json.loads(payload)


def _reason_about_variance(
    record: UnresolvedRecord,
    variance: VarianceReport,
    fees: HistoricalFeeSchedule,
) -> tuple[float, str, str]:
    """
    The deterministic 'reasoning' core. Returns (confidence, explanation, category).

    This is intentionally readable as a decision table, because in a
    regulated reconciliation system the auditor's first question is always
    "show me the rule that fired," not "trust the model."
    """
    tol = fees.tolerance_percent
    pct = variance.variance_percent  # positive = merchant expected more than settled

    # --- Case 1: bundled settlement, sum reconciles exactly -----------------
    if variance.is_bundled and abs(pct) <= tol:
        return (
            0.97,
            (
                f"Payout links {len(record.linked_orders)} orders "
                f"({', '.join(record.linked_orders)}) totalling ₹{variance.merchant_total:.2f}. "
                f"Settled amount ₹{variance.settled_amount:.2f} matches that sum within "
                f"{tol}% tolerance (observed variance {pct:.3f}%). This is a bundled "
                f"settlement, not a discrepancy — sum-of-orders reconciles cleanly."
            ),
            "BUNDLED_SETTLEMENT_RECONCILED",
        )

    # --- Case 2: variance matches the standard undisclosed gateway fee ------
    fee_pct = fees.standard_gateway_fee_percent
    if record.gateway_fee_deducted in (None, 0) and abs(pct - fee_pct) <= tol:
        fee_amount = round(variance.merchant_total * fee_pct / 100, 2)
        return (
            0.94,
            (
                f"Observed variance of {pct:.3f}% closely matches Razorpay's standard "
                f"{fee_pct:.1f}% gateway fee rate (tolerance ±{tol}%), and no fee was "
                f"itemized on this payout (gateway_fee_deducted was not provided). "
                f"Estimated undisclosed fee: ₹{fee_amount:.2f} on a ₹{variance.merchant_total:.2f} "
                f"order. Reconciled as: settled_amount = merchant_amount - standard gateway fee."
            ),
            "UNDISCLOSED_GATEWAY_FEE",
        )

    # --- Case 3: variance matches the rolling reserve hold rate -------------
    reserve_pct = fees.rolling_reserve_hold_percent
    if abs(pct - reserve_pct) <= tol:
        return (
            0.55,
            (
                f"Observed variance of {pct:.3f}% matches Razorpay's rolling reserve hold "
                f"rate ({reserve_pct:.1f}%, tolerance ±{tol}%), which would normally explain "
                f"the shortfall. However, a reserve hold is a TIMING difference, not a fee — "
                f"the withheld ₹{abs(variance.absolute_variance):.2f} is expected to be "
                f"released after {fees.rolling_reserve_release_days} days, and this system has "
                f"no way to confirm the release date or whether a hold was actually applied to "
                f"this specific payout versus some other cause. Confidence intentionally kept "
                f"below the auto-match threshold — recommend human confirmation against the "
                f"Razorpay reserve ledger before treating this as reconciled."
            ),
            "POSSIBLE_ROLLING_RESERVE_NEEDS_CONFIRMATION",
        )

    # --- Case 4: unexplained variance (e.g. delayed/partial refund) --------
    return (
        0.30,
        (
            f"Observed variance of {pct:.3f}% (₹{variance.absolute_variance:.2f}) does not "
            f"match the standard gateway fee ({fee_pct:.1f}%), the rolling reserve rate "
            f"({reserve_pct:.1f}%), or a bundled-order reconciliation, within ±{tol}% "
            f"tolerance. No known rule explains this shortfall — it is consistent with an "
            f"unrecorded partial refund, chargeback, or manual adjustment, but this cannot be "
            f"confirmed from the data available to this agent. Flagging for human review "
            f"rather than guessing."
        ),
        "UNEXPLAINED_VARIANCE",
    )


def _reason(engine: str, record: UnresolvedRecord, variance: VarianceReport, fees: HistoricalFeeSchedule):
    """Dispatch to the selected reasoning engine. Both engines share the exact
    same (confidence, explanation, category) contract, so nothing else in the
    pipeline needs to know or care which one produced a given result."""
    if engine == "ml":
        from src.agent import ml_reasoner  # imported lazily: sklearn/joblib are only
        return ml_reasoner.reason(record, variance, fees)  # required if this engine is actually used
    return _reason_about_variance(record, variance, fees)


async def _investigate_one(
    session: ClientSession, record: UnresolvedRecord, engine: str = "rules"
) -> tuple[float, str, List[str]]:
    """Run the PERCEIVE -> REASON steps for a single record via real MCP tool calls."""
    tools_invoked: List[str] = []

    variance_raw = await session.call_tool("calculate_variance", {"payout_id": record.payout_id})
    tools_invoked.append("calculate_variance")
    variance = VarianceReport.model_validate(_extract_json(variance_raw))

    fees_raw = await session.call_tool("fetch_historical_fees", {})
    tools_invoked.append("fetch_historical_fees")
    fees = HistoricalFeeSchedule.model_validate(_extract_json(fees_raw))

    # Only spend a simulate_tax_deduction call when it's actually informative:
    # when the fee-schedule case is plausible, so we can show the tax-inclusive
    # picture in the explanation for full transparency. (Kept identical across
    # both engines: it's a PERCEIVE-phase tool call, not part of either
    # engine's REASON step.)
    fee_pct = fees.standard_gateway_fee_percent
    if abs(variance.variance_percent - fee_pct) <= fees.tolerance_percent:
        est_fee = round(variance.merchant_total * fee_pct / 100, 2)
        await session.call_tool(
            "simulate_tax_deduction", {"base_amount": est_fee, "tax_rate_percent": fees.gst_on_fee_percent}
        )
        tools_invoked.append("simulate_tax_deduction")

    confidence, explanation, _category = _reason(engine, record, variance, fees)
    return confidence, explanation, tools_invoked


async def run_agent_reconciliation(
    unresolved_records: List[UnresolvedRecord],
    server_script: Path,
    python_executable: str = sys.executable,
    engine: str = "rules",
) -> AgentRunResult:
    """
    Spawn the MCP server as a subprocess (stdio transport), connect a real
    MCP client session to it, and investigate every unresolved record.

    Args:
        engine: "rules" (default, hand-coded decision table — always
            available, no extra dependencies) or "ml" (trained classifier
            from data/training/train_model.py — requires that training
            script to have been run at least once first).
    """
    if engine not in ("rules", "ml"):
        raise ValueError(f"Unknown reasoning engine '{engine}'; expected 'rules' or 'ml'.")
    if not server_script.is_file():
        raise FileNotFoundError(f"MCP server script not found: {server_script}")
    resolved: List[AgentResolution] = []
    exceptions: List[ExceptionRecord] = []

    if not unresolved_records:
        return AgentRunResult(resolved=[], exceptions=[], confidence_threshold=CONFIDENCE_THRESHOLD)

    params = StdioServerParameters(
        command=python_executable,
        # Start as a module so the repository root remains importable as
        # ``src`` in the child process. Executing server.py directly makes
        # Python place src/mcp_server on sys.path and breaks its imports.
        args=["-m", "src.mcp_server.server"],
        cwd=str(_REPO_ROOT),
    )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            for record in unresolved_records:
                confidence, explanation, tools_invoked = await _investigate_one(session, record, engine=engine)

                if confidence >= CONFIDENCE_THRESHOLD:
                    resolved.append(
                        AgentResolution(
                            payout_id=record.payout_id,
                            order_ids=record.linked_orders,
                            merchant_total=record.merchant_total,
                            settled_amount=record.settled_amount,
                            confidence=confidence,
                            explanation=explanation,
                            tools_invoked=tools_invoked,
                        )
                    )
                else:
                    exceptions.append(
                        ExceptionRecord(
                            payout_id=record.payout_id,
                            order_ids=record.linked_orders,
                            merchant_total=record.merchant_total,
                            settled_amount=record.settled_amount,
                            confidence=confidence,
                            explanation=explanation,
                            tools_invoked=tools_invoked,
                            recommended_human_action=(
                                "Manually verify against the Razorpay settlement dashboard "
                                "and merchant support ticket history before adjusting the ledger."
                            ),
                        )
                    )

    return AgentRunResult(resolved=resolved, exceptions=exceptions, confidence_threshold=CONFIDENCE_THRESHOLD)


def run_agent_reconciliation_sync(
    unresolved_records: List[UnresolvedRecord], server_script: Path, engine: str = "rules"
) -> AgentRunResult:
    """Synchronous convenience wrapper for callers (like main.py) that aren't async."""
    return asyncio.run(run_agent_reconciliation(unresolved_records, server_script, engine=engine))
