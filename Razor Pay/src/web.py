"""Local dashboard for the settlement reconciliation demo.

Run ``python -m src.web`` after installing requirements. The server is
intentionally standard-library only so the dashboard adds no deployment
dependency or data egress. It binds to 127.0.0.1 and opens the browser.
"""
from __future__ import annotations

import json
import subprocess
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
HOST = "127.0.0.1"
PORT = 8765


def _read_json(name: str, default: object) -> object:
    path = DATA_DIR / name
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _dashboard_payload() -> dict:
    ledger = _read_json("reconciled_ledger.json", {})
    exceptions = _read_json("honest_exceptions.json", [])
    if not isinstance(ledger, dict):
        ledger = {}
    summary = ledger.get("summary", {})
    deterministic = ledger.get("deterministic_matches", [])
    ai_resolved = ledger.get("ai_resolved_matches", [])
    total = summary.get("total_reconciliation_cases", summary.get("total_payouts", 0))
    resolved = len(deterministic) + len(ai_resolved)
    return {
        "ready": bool(summary),
        "summary": {**summary, "resolved_count": resolved, "resolution_rate": (resolved / total if total else 0)},
        "exceptions": exceptions if isinstance(exceptions, list) else [],
        "recent_resolutions": (deterministic + ai_resolved)[-8:],
    }


class DashboardHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/api/dashboard":
            self._send_json(_dashboard_payload())
            return
        if urlparse(self.path).path != "/":
            self.send_error(404)
            return
        body = _PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/run":
            self.send_error(404)
            return
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "src.main"], cwd=ROOT, capture_output=True,
                text=True, timeout=90, check=True,
            )
            self._send_json({"ok": True, "output": completed.stdout, "dashboard": _dashboard_payload()})
        except subprocess.CalledProcessError as error:
            self._send_json({"ok": False, "error": error.stderr or error.stdout}, 500)
        except subprocess.TimeoutExpired:
            self._send_json({"ok": False, "error": "Reconciliation exceeded the 90-second safety limit."}, 504)

    def log_message(self, _format: str, *_args: object) -> None:
        """Keep browser requests out of the clean terminal experience."""


_PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Razorpay | Reconciliation Control Center</title><style>
:root{--blue:#0b72e7;--navy:#072654;--ink:#172b4d;--muted:#6b7a90;--line:#e7edf5;--bg:#f6f9fd;--green:#168c5b;--orange:#d97917}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}.nav{height:70px;background:#fff;display:flex;align-items:center;padding:0 max(24px,6vw);border-bottom:1px solid var(--line);gap:18px}.brand{font-weight:800;color:var(--blue);font-size:23px;letter-spacing:-1px}.brand span{color:#1d375f}.badge{background:#e9f3ff;color:var(--blue);padding:5px 9px;border-radius:20px;font-size:12px;font-weight:700}.wrap{max-width:1180px;margin:0 auto;padding:52px 24px}.hero{display:flex;justify-content:space-between;align-items:end;gap:30px;margin-bottom:34px}.eyebrow{color:var(--blue);font-size:12px;font-weight:800;letter-spacing:1.2px;text-transform:uppercase}.hero h1{font-size:clamp(29px,4vw,45px);letter-spacing:-1.8px;line-height:1.08;margin:9px 0 12px}.hero p{max-width:650px;color:var(--muted);line-height:1.6;margin:0}.run{border:0;background:var(--blue);color:#fff;border-radius:8px;padding:14px 19px;font-weight:750;font-size:14px;cursor:pointer;box-shadow:0 8px 20px #0b72e73c;white-space:nowrap}.run:disabled{opacity:.65;cursor:wait}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.card,.panel{background:#fff;border:1px solid var(--line);border-radius:12px;box-shadow:0 5px 12px #1f47730a}.card{padding:21px}.label{font-size:12px;color:var(--muted);font-weight:700}.metric{font-size:30px;font-weight:800;margin:8px 0 4px}.note{font-size:12px;color:var(--muted)}.panels{display:grid;grid-template-columns:1.2fr .8fr;gap:16px;margin-top:16px}.panel{padding:24px}.panel h2{font-size:17px;margin:0 0 17px}.row{display:flex;justify-content:space-between;gap:16px;padding:13px 0;border-top:1px solid var(--line);font-size:14px}.row:first-of-type{border-top:0}.id{font-weight:750}.right{text-align:right}.status{font-size:11px;font-weight:800;padding:4px 7px;border-radius:12px;color:var(--green);background:#e7f8f0}.exception{color:var(--orange);background:#fff4e6}.empty{color:var(--muted);padding:15px 0;font-size:14px}.log{display:none;margin-top:16px;background:#08234d;color:#dcecff;padding:18px;border-radius:10px;white-space:pre-wrap;font:12px/1.5 ui-monospace,monospace;max-height:250px;overflow:auto}@media(max-width:800px){.hero{align-items:start;flex-direction:column}.grid{grid-template-columns:repeat(2,1fr)}.panels{grid-template-columns:1fr}}@media(max-width:430px){.grid{grid-template-columns:1fr}.wrap{padding:32px 16px}}
</style></head><body><nav class="nav"><div class="brand">razorpay<span>.</span></div><div class="badge">AI BUILDATHON DEMO</div></nav><main class="wrap"><section class="hero"><div><div class="eyebrow">Settlement reconciliation</div><h1>Finance operations, with an honest safety net.</h1><p>Exact matches clear automatically. Ambiguous settlements are investigated through a local MCP boundary and only high-confidence cases reach the ledger.</p></div><button class="run" id="run" onclick="runPipeline()">Run reconciliation</button></section><section class="grid"><div class="card"><div class="label">RECONCILIATION CASES</div><div class="metric" id="cases">—</div><div class="note">Payouts and missing-payout orders</div></div><div class="card"><div class="label">AUTO-RESOLVED</div><div class="metric" id="resolved">—</div><div class="note">Deterministic + AI-assisted</div></div><div class="card"><div class="label">RESOLUTION RATE</div><div class="metric" id="rate">—</div><div class="note">Never includes unverified guesses</div></div><div class="card"><div class="label">HUMAN REVIEW</div><div class="metric" id="exceptions">—</div><div class="note">Honest exception queue</div></div></section><section class="panels"><div class="panel"><h2>Recent resolved settlements</h2><div id="resolutions" class="empty">Run the reconciliation to populate the ledger.</div></div><div class="panel"><h2>Exception queue</h2><div id="exceptionList" class="empty">No results yet.</div></div></section><pre class="log" id="log"></pre></main><script>
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));function money(n){return new Intl.NumberFormat('en-IN',{style:'currency',currency:'INR',maximumFractionDigits:2}).format(n||0)}function render(d){const s=d.summary||{};document.querySelector('#cases').textContent=s.total_reconciliation_cases??s.total_payouts??'—';document.querySelector('#resolved').textContent=s.resolved_count??'—';document.querySelector('#rate').textContent=s.total_reconciliation_cases?(s.resolution_rate*100).toFixed(1)+'%':'—';document.querySelector('#exceptions').textContent=(d.exceptions||[]).length;document.querySelector('#resolutions').innerHTML=(d.recent_resolutions||[]).length?d.recent_resolutions.map(x=>`<div class="row"><div><div class="id">${esc(x.payout_id)}</div><div class="note">${esc((x.order_ids||[]).join(', '))}</div></div><div class="right"><span class="status">${x.match_type==='AI_RESOLVED'?'AI VERIFIED':'EXACT MATCH'}</span><div class="note">${money(x.settled_amount)}</div></div></div>`).join(''):'<div class="empty">Run the reconciliation to populate the ledger.</div>';document.querySelector('#exceptionList').innerHTML=(d.exceptions||[]).length?d.exceptions.slice(0,5).map(x=>`<div class="row"><div><div class="id">${esc(x.payout_id)}</div><div class="note">${esc(x.explanation).slice(0,95)}…</div></div><div class="right"><span class="status exception">${Math.round((x.confidence||0)*100)}% confidence</span></div></div>`).join(''):'<div class="empty">No exceptions waiting for review.</div>'}async function refresh(){render(await fetch('/api/dashboard').then(r=>r.json()))}async function runPipeline(){let b=document.querySelector('#run'),log=document.querySelector('#log');b.disabled=true;b.textContent='Reconciling…';log.style.display='block';log.textContent='Running the local pipeline…';try{let r=await fetch('/api/run',{method:'POST'}),d=await r.json();if(!r.ok)throw Error(d.error||'Pipeline failed.');log.textContent=d.output;render(d.dashboard)}catch(e){log.textContent='Error: '+e.message}finally{b.disabled=false;b.textContent='Run reconciliation'}}refresh();
</script></body></html>'''


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    url = f"http://{HOST}:{PORT}"
    print(f"Dashboard running at {url} (press Ctrl+C to stop)")
    webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
