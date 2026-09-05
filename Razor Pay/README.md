# RazorRecon — Automated Settlement Reconciliation Engine

**Razorpay AI Buildathon · Track 04: AI Finance Controller**

RazorRecon is a local-first reconciliation demo for merchant sales and Razorpay payout records. It clears only exact, explainable settlements automatically; ambiguous cases are investigated through a constrained MCP server and are sent to a human-review queue whenever confidence is below 90%.

No API keys, cloud services, or external financial calls are required.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white) ![MCP](https://img.shields.io/badge/MCP-local%20stdio-5A45FF) ![Safety](https://img.shields.io/badge/Auto--resolve-90%25%20minimum-168C5B)

## What it does

1. Reads merchant sales and payout records.
2. Deterministically clears only strict one-order, exact-amount matches using `Decimal` money comparisons.
3. Sends every other case through a local MCP server with narrowly scoped read-only tools.
4. Explains fee-shaped and bundled settlements with a reproducible rule engine (or optional ML classifier).
5. Writes a reconciled ledger, a review queue, and an audit trail.

The dashboard is a convenient local view of the same pipeline; it does not send data anywhere.

## Run it locally

### Windows (PowerShell)

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL> RazorRecon
cd RazorRecon
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python data\generate_data.py
python -m src.web
```

The final command opens **http://127.0.0.1:8765** in the default browser with the Razorpay-style dashboard. Click **Run reconciliation** to execute the full pipeline; leave that terminal open while using the UI and press `Ctrl+C` to stop it.

If PowerShell blocks virtual-environment activation, run this once in that terminal and repeat the activation command:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### macOS / Linux

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL> RazorRecon
cd RazorRecon
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python data/generate_data.py
python -m src.web
```

## Terminal workflow

Use this when you want the artifacts without opening the dashboard:

```bash
python data/generate_data.py
python -m src.main
python -m pytest tests -v
```

Generated local artifacts:

- `data/reconciled_ledger.json` — exact and AI-assisted resolutions
- `data/honest_exceptions.json` — unresolved cases that require human confirmation
- `data/audit_trail.md` — human-readable evidence and decision trail
- `data/deterministic_result.json` — the restricted hand-off available to the MCP server

## Safety model

| Layer | Decision | Guardrail |
|---|---|---|
| Deterministic matcher | Exact single-order payment | Uses `Decimal`, zero tolerance |
| MCP boundary | Provides evidence to the agent | Read-only, local stdio, no raw-file access for the agent |
| Reasoning engine | Explains eligible cases | 90% minimum confidence to write a resolution |
| Exception queue | Retains uncertainty | Reserve holds and unexplained variances are never silently cleared |

Additional input controls reject duplicate order IDs, duplicate payout IDs, duplicate orders within a payout, and an order referenced by more than one payout. These are surfaced as unresolved rather than risking double counting.

## Architecture

```text
merchant_sales.json + razorpay_payouts.json
                    │
                    ▼
        Deterministic matcher (exact 1:1 only)
                    │
                    ▼
        deterministic_result.json (restricted hand-off)
                    │
                    ▼
          Local MCP server (read-only tools)
                    │
                    ▼
       Rules engine / optional ML classifier
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 reconciled_ledger.json  honest_exceptions.json
          │
          ▼
      audit_trail.md + local dashboard
```

## Optional ML engine

The rules engine is the default because it is fully reproducible and needs no model artifact. An optional `RandomForestClassifier` uses the same feature extractor as training, and policy still blocks auto-resolution for reserve-hold and unexplained categories.

```bash
python -m data.training.generate_training_batches
python -m data.training.train_model
python -m src.main --engine ml
```

Review `data/training/model_report.md` before relying on a retrained model.

## Verify before submission

```bash
python -m pytest tests -v
python data/generate_data.py
python -m src.deterministic.matcher
python -m src.main
```

The project requires Python 3.10+ and installs all runtime and test dependencies from `requirements.txt`.

For the Buildathon demo narration, use [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md).

## Project layout

```text
src/
  main.py                   Full pipeline orchestration
  web.py                    Local Razorpay-style dashboard
  models.py                 Shared Pydantic schemas
  deterministic/matcher.py  Exact-match and input-integrity layer
  mcp_server/server.py      Local read-only MCP tools
  agent/reconciler.py       Confidence-gated rules/ML dispatch
data/
  generate_data.py          Reproducible synthetic demo data
  training/                 ML training data and trainer
tests/                      Unit tests for matcher, agent, MCP, and ML policy
```

## Important note

This is a buildathon demo using synthetic data and a static local fee schedule. It is not connected to Razorpay production systems and must not be used to make production ledger changes without appropriate security, data governance, and human approval.
