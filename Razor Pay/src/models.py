"""
src/models.py
==============

Single source of truth for every data shape used in the reconciliation
pipeline. Every other module (deterministic matcher, MCP server, AI agent,
main orchestrator) imports its types from here rather than redefining them.

Why this matters for a fintech system
--------------------------------------
In a regulated financial system, an inconsistent schema between the layer
that reads bank data and the layer that reasons about it is a silent
correctness bug waiting to happen (e.g. one layer treating a missing fee
as `0.0` and another treating it as `None`). Centralizing the models means
a schema change is a one-file review, not an N-file hunt.

See AGENTS.md for the change-control rule that governs edits to this file.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Raw input records (mirror the on-disk JSON schemas exactly)
# ---------------------------------------------------------------------------


class MerchantSale(BaseModel):
    """A single row from the merchant's internal 'what I expect to be paid' ledger."""

    order_id: str = Field(..., description="Merchant-side unique order identifier.")
    timestamp: str = Field(..., description="ISO-8601 order timestamp.")
    amount: float = Field(..., gt=0, description="Amount the merchant expects to receive, in INR.")
    customer_id: str = Field(..., description="Merchant-side customer identifier.")


class RazorpayPayout(BaseModel):
    """A single row from the actual bank settlement file Razorpay produces."""

    payout_id: str = Field(..., description="Razorpay-side unique payout identifier.")
    linked_orders: List[str] = Field(
        ..., min_length=1, description="Order id(s) this payout claims to settle."
    )
    settled_amount: float = Field(..., ge=0, description="Actual amount settled to the merchant's bank account.")
    gateway_fee_deducted: Optional[float] = Field(
        default=None,
        description="Fee Razorpay explicitly itemized for this payout. "
        "None means no line item was provided (a common source of discrepancy).",
    )
    tax_deducted: Optional[float] = Field(
        default=None,
        description="Tax (e.g. GST on fees) explicitly itemized for this payout. None if not itemized.",
    )

    @field_validator("linked_orders")
    @classmethod
    def _no_duplicate_orders(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            raise ValueError("linked_orders must not contain duplicate order ids")
        return v


# ---------------------------------------------------------------------------
# Deterministic matcher output (Phase 3)
# ---------------------------------------------------------------------------


class MatchedRecord(BaseModel):
    """A record the deterministic matcher resolved with zero ambiguity."""

    order_ids: List[str]
    payout_id: str
    merchant_total: float
    settled_amount: float
    match_type: str = "DETERMINISTIC_EXACT"


class UnresolvedRecord(BaseModel):
    """
    A payout that the deterministic layer could NOT confidently clear.
    This is the ONLY object type the MCP server is allowed to expose to the
    AI agent — the agent never sees the raw source JSON files directly.
    """

    payout_id: str
    linked_orders: List[str]
    merchant_amounts: Dict[str, float] = Field(
        ..., description="order_id -> merchant-expected amount, for every linked order."
    )
    settled_amount: float
    gateway_fee_deducted: Optional[float] = None
    tax_deducted: Optional[float] = None
    reason: str = Field(..., description="Why the deterministic layer punted on this record.")

    @property
    def merchant_total(self) -> float:
        return round(sum(self.merchant_amounts.values()), 2)

    @property
    def raw_variance(self) -> float:
        return round(self.merchant_total - self.settled_amount, 2)


class DeterministicMatchResult(BaseModel):
    """Full output of Phase 3."""

    matched: List[MatchedRecord]
    unresolved: List[UnresolvedRecord]
    total_merchant_orders: int
    total_payouts: int


# ---------------------------------------------------------------------------
# MCP tool payloads (Phase 4) — these ARE the contract between the AI agent
# and the secure data-access layer. Changing these shapes is a breaking
# change to the "USB-C port"; see AGENTS.md before editing.
# ---------------------------------------------------------------------------


class VarianceReport(BaseModel):
    """Result of the `calculate_variance` MCP tool."""

    payout_id: str
    merchant_total: float
    settled_amount: float
    absolute_variance: float
    variance_percent: float = Field(..., description="(merchant_total - settled_amount) / merchant_total * 100")
    is_bundled: bool = Field(..., description="True if more than one order is linked to this payout.")


class HistoricalFeeSchedule(BaseModel):
    """Result of the `fetch_historical_fees` MCP tool. A static, local, known-rate table —
    NOT a network call. This models Razorpay's publicly documented standard rate card."""

    standard_gateway_fee_percent: float
    gst_on_fee_percent: float
    rolling_reserve_hold_percent: float
    rolling_reserve_release_days: int
    tolerance_percent: float = Field(
        ..., description="Acceptable drift when comparing a variance against one of the rates above."
    )


class TaxSimulationResult(BaseModel):
    """Result of the `simulate_tax_deduction` MCP tool."""

    base_amount: float
    tax_rate_percent: float
    simulated_tax_amount: float
    net_of_tax: float


# ---------------------------------------------------------------------------
# AI agent output (Phase 5)
# ---------------------------------------------------------------------------


class AgentResolution(BaseModel):
    """A record the AI agent resolved with confidence >= the configured threshold."""

    payout_id: str
    order_ids: List[str]
    merchant_total: float
    settled_amount: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    tools_invoked: List[str]
    match_type: str = "AI_RESOLVED"


class ExceptionRecord(BaseModel):
    """A record NOT resolved by either layer. Confidence fell below threshold —
    this is the system being honest about the limits of its own reasoning."""

    payout_id: str
    order_ids: List[str]
    merchant_total: float
    settled_amount: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    tools_invoked: List[str]
    recommended_human_action: str


class AgentRunResult(BaseModel):
    """Full output of Phase 5."""

    resolved: List[AgentResolution]
    exceptions: List[ExceptionRecord]
    confidence_threshold: float
