# AGENTS.md

This file constrains any agent — human or AI — making changes to this
repository. It is the structural contract for the Automated Settlement
Reconciliation Engine (Razorpay AI Buildathon, Track 04: AI Finance
Controller). Read this before touching `src/models.py`, `src/mcp_server/`,
or `src/agent/`.

## 1. Non-negotiable system invariants

1. **Zero financial hallucination.** No code path may write a resolved
   match to `data/reconciled_ledger.json` unless it came from either (a)
   the deterministic matcher's exact-amount rule, or (b) the AI agent with
   `confidence >= CONFIDENCE_THRESHOLD` (currently `0.90`, defined once in
   `src/agent/reconciler.py`). Lowering this threshold, or adding a code
   path that bypasses it, is a breaking change to the system's core
   promise and requires sign-off, not a quiet edit.
2. **The MCP server is the only door.** The AI agent must never read
   `data/merchant_sales.json` or `data/razorpay_payouts.json` directly.
   It only sees what `src/mcp_server/server.py` chooses to expose via
   `list_unresolved_records`. If a future record type needs to be exposed
   to the agent, add a new tool with the same validation/logging discipline
   as the existing three — do not add a generic "read any file" tool.
3. **No outbound network calls from the MCP server or the agent.**
   `fetch_historical_fees` is a static local lookup table, not a live API
   call, by design — it models Razorpay's published rate card, not a
   real-time endpoint. If a genuinely live fee-lookup integration is added
   later, it must be a new, explicitly-named tool (e.g.
   `fetch_live_fee_schedule`) so the offline/online distinction stays
   visible in the tool list, not silently swapped in place.
4. **Every tool call is logged.** `src/mcp_server/server.py`'s `_log()`
   helper writes a structured, timestamped line to stderr for every tool
   invocation. Do not remove or silence this — it is the raw material the
   audit trail in `src/main.py` is built to be traceable against.

## 2. Schema change control

- `src/models.py` is the single source of truth for every data shape in
  the pipeline. **Never redefine a model shape locally in another file.**
- Any change to `UnresolvedRecord`, `VarianceReport`,
  `HistoricalFeeSchedule`, or `TaxSimulationResult` is a change to the
  MCP tool contract between the server and the agent. Before merging such
  a change:
  1. Update `src/models.py`.
  2. Update the corresponding tool signature/return in
     `src/mcp_server/server.py`.
  3. Update `src/agent/reconciler.py`'s parsing of that tool's response.
  4. Re-run `pytest tests/ -v` — all three test files touch this
     contract from a different angle (matcher, server, agent) and a
     silent mismatch between layers should fail at least one of them.
- Field renames are breaking changes for anything that reads
  `data/deterministic_result.json` as a handoff file. Do not rename a
  field without regenerating that file (`python -m src.deterministic.matcher`)
  before the next run.

## 3. Isolated test environment

- Tests live in `tests/` and run against `src/` via absolute imports
  (`from src.deterministic.matcher import ...`), assuming the repo root
  is on `PYTHONPATH` / is the working directory. Run them as:
  ```
  python -m pytest tests/ -v
  ```
  from the repository root — not `cd tests && pytest`.
- `tests/test_matcher.py` and `tests/test_mcp_server.py`/`test_agent.py`
  never touch the real `data/*.json` files. They build their own
  temporary fixtures (`tmp_path`) or monkeypatch server-internal state.
  This is deliberate: the synthetic dataset in `data/` is meant to be
  regenerable/replaceable without breaking the test suite, and the test
  suite is meant to be runnable without a prior `python -m src.main` run.
- `tests/test_agent.py` unit-tests `_reason_about_variance` directly and
  does NOT spin up the MCP stdio transport — that would make the unit
  suite slow and flaky in CI. Live transport correctness is instead
  exercised by actually running `python -m src.main`, which is the
  recommended CI smoke-test step (see below).

## 4. CI/CD expectations

A CI pipeline for this repo should run, in order, on every change:

1. `pip install -r requirements.txt`
2. `python -m pytest tests/ -v` — must be 100% green before proceeding.
3. `python data/generate_data.py` — regenerate the seeded synthetic
   dataset (deterministic; should produce byte-identical output given
   the fixed `SEED = 42`, so this step also catches accidental
   nondeterminism regressions).
4. `python -m src.deterministic.matcher` — smoke-test Phase 3 alone.
5. `python -m src.main` — full pipeline smoke test, exercising the real
   MCP client/server stdio round trip end to end. This step is the only
   place actual MCP transport correctness is verified; do not remove it
   from CI even though it's slower than the unit tests.
6. Fail the build if `data/audit_trail.md` was not regenerated, or if
   `data/honest_exceptions.json` is missing — both are required
   deliverables of the pipeline, not optional artifacts.
7. (Optional, only if `--engine ml` is in use anywhere downstream)
   `python -m data.training.generate_training_batches && python -m data.training.train_model`,
   then check `data/training/model_report.md`'s held-out accuracy hasn't
   regressed, then `python -m src.main --engine ml` as a second smoke test
   alongside step 5's default `--engine rules` run.

## 5. The ML reasoning engine (`--engine ml`)

`src/agent/ml_reasoner.py` is a second, optional reasoning engine trained
on `data/training/`'s 10 dummy batches. It implements the exact same
`(confidence, explanation, category) = reason(record, variance, fees)`
contract as the rule-based engine in `src/agent/reconciler.py`, selected
via `python -m src.main --engine ml`. Rules of the road:

1. **The rule-based engine is and stays the default.** `--engine` defaults
   to `"rules"`. The rule engine has zero extra dependencies and a fixed,
   always-reproducible decision table; it is the safe fallback if the
   model artifact is missing, stale, or under dispute. Do not flip the
   default without an explicit decision to do so.
2. **Confidence caps are a policy decision, not a training artifact.**
   `ml_reasoner.AUTO_RESOLVABLE_CATEGORIES` is a hard allow-list. A
   category outside it (currently `ROLLING_RESERVE_HOLD`,
   `UNEXPLAINED_VARIANCE`, `FEE_PLUS_RESERVE_COMBINED`) is capped to a
   sub-threshold confidence NO MATTER HOW CONFIDENT THE MODEL IS, because
   those categories represent timing differences or genuinely ambiguous
   evidence this system cannot independently verify. If you retrain the
   model and it becomes highly accurate on `FEE_PLUS_RESERVE_COMBINED`,
   that is a reason to consider it for the allow-list deliberately — not a
   reason for the cap to silently stop applying.
3. **Feature extraction must not drift between training and serving.**
   `src/agent/features.py` is imported by BOTH `data/training/train_model.py`
   and `src/agent/ml_reasoner.py`. Never reimplement feature computation
   in either file separately — this is exactly the train/serve skew bug
   class that quietly produces wrong predictions in production while all
   tests still pass.
4. **Retraining procedure.** To regenerate training data and retrain:
   ```
   python -m data.training.generate_training_batches   # 10 seeded, reproducible batches
   python -m data.training.train_model                 # trains + writes model_report.md
   ```
   The held-out test set is two ENTIRE batches (indices 8-9 by default in
   `TEST_BATCH_INDICES`), not a random row split — read
   `data/training/model_report.md`'s held-out accuracy and confusion
   matrix before trusting a retrained model, and do not merge a retrain
   that drops held-out accuracy meaningfully below the last committed
   report without understanding why.
5. **The model artifact is a build output, not hand-edited.** Never edit
   `data/training/edge_case_model.joblib` directly. If you need to change
   what the model predicts, change the training data generator or the
   model hyperparameters in `train_model.py`, then retrain.
6. `tests/test_ml_reasoner.py` is skipped (not failed) if the model
   artifact doesn't exist, so a fresh checkout's test suite stays green
   before training has run. CI should still run the retraining procedure
   above before the full test suite to get real coverage of this engine,
   not rely on the skip.

## 6. Dependency upgrade policy

- `mcp` is pinned to `1.9.4` in `requirements.txt`. Do not bump to `mcp>=2`
  without first inspecting the new package's real `FastMCP`/`MCPServer`
  API via `inspect.signature(...)` in a scratch shell — the 2.x line
  renamed core classes and changed several signatures. Guessing at the
  new API instead of verifying it against the installed package is
  exactly the "hallucinated API call" failure mode this project was
  built to avoid; don't reintroduce it during a dependency bump.
- If a future agent (AI or human) adds `langchain`/`langgraph` per the
  original brief's suggestion, see the rationale and integration point
  documented at the bottom of `requirements.txt` first.

## 7. Style and structure

- Strict typing throughout: every function signature carries real type
  hints, and cross-module data always flows through a `pydantic.BaseModel`
  from `src/models.py`, never a bare `dict`.
- Do not add `from __future__ import annotations` to
  `src/mcp_server/server.py`. FastMCP's tool inspector reads live
  parameter annotations at decoration time; postponed evaluation turns
  them into strings and breaks tool registration (this is a real bug
  that was hit and fixed during this repo's initial build — see the
  module docstring history / git blame before "fixing" it again).
- Money is compared with `decimal.Decimal`, never raw `float ==`, inside
  `src/deterministic/matcher.py`. Keep it that way for any new exact-match
  logic.
