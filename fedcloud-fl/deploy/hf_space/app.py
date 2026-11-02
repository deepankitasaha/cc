"""Hugging Face Spaces entry point: the FedAsync aggregation server on one public HTTPS port (7860).

Wraps fedcloud.async_server so the Space keeps serving between demo runs:
  GET  /                    live dashboard (auto-refresh)
  GET  /health, /model      unchanged FedAsync protocol
  POST /update              unchanged FedAsync protocol
  GET  /status              JSON progress of the current run
  GET  /results             list of finished runs; /results/<n>.json returns one log
  GET  /reset               end the current run early (clients get 410) and start a fresh one
When a run reaches its update budget the clients receive 410, the log is kept in memory and a
new run starts automatically, so the next demo needs no restart.
"""
from __future__ import annotations

import html
import json
import os
import threading
import time
from argparse import Namespace
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse

from fedcloud.async_server import AsyncState, finalize_log, make_handler
from fedcloud.data import load_npz

CFG = Namespace(rounds=int(os.environ.get("ROUNDS", 30)), clients=int(os.environ.get("CLIENTS", 5)),
                alpha=0.4, staleness_a=0.5, epochs=1, lr=1e-3, seed=0, tag="hf-space")
PORT = int(os.environ.get("PORT", 7860))
TEST = load_npz(os.environ.get("TEST", "server_test.npz"))
RUNS: list[dict] = []


class Current:
    """Forwards every attribute read/write to the live run, so the stock handler always talks to it."""

    def __init__(self):
        self.swap()

    def swap(self):
        object.__setattr__(self, "s", AsyncState(CFG, TEST))
        object.__setattr__(self, "created", time.time())

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "s"), k)

    def __setattr__(self, k, v):
        setattr(object.__getattribute__(self, "s"), k, v)


CUR = Current()


def supervise():
    while True:
        s, created = CUR.s, CUR.created
        s.done.wait()
        time.sleep(1.0)  # let in-flight clients receive 410
        with s.lock:
            if s.version:
                log = finalize_log(s, created)
                log["run"] = len(RUNS)
                log["complete"] = s.version >= s.budget
                RUNS.append(log)
                print(f"[hf-space] run {log['run']} finished: participation={log['participation']} "
                      f"mean staleness={log['mean_staleness']:.2f}", flush=True)
            CUR.swap()


def status() -> dict:
    s = CUR.s
    with s.lock:
        last = s.log["history"][-1] if s.log["history"] else {}
        return {"run": len(RUNS), "version": s.version, "budget": s.budget, "clients": CFG.clients,
                "started": s.t0 is not None, "participation": {str(k): v for k, v in sorted(s.participation.items())},
                "mean_staleness": round(sum(s.staleness) / max(1, len(s.staleness)), 2),
                "last_eval": last, "history": s.log["history"],
                "finished_runs": [{"run": r["run"], "complete": r["complete"], "updates": r["history"][-1]["updates"],
                                   "auprc": r["history"][-1]["auprc"], "total_wall_s": r["total_wall_s"]}
                                  for r in RUNS]}


def page() -> bytes:
    st = status()
    rows = "".join(f"<tr><td>{h['round']}</td><td>{h['updates']}</td><td>{h['wall_s']:.1f}</td>"
                   f"<td>{h['bytes_total'] / 1e6:.3f}</td><td>{h['auprc']:.4f}</td><td>{h['f1']:.4f}</td>"
                   f"<td>{h['roc_auc']:.4f}</td></tr>" for h in reversed(st["history"]))
    part = ", ".join(f"client {k}: {v}" for k, v in st["participation"].items()) or "waiting for clients…"
    done = "".join(f"<li><a href='results/{r['run']}.json'>run {r['run']}</a> — {r['updates']} updates, "
                   f"AUPRC {r['auprc']:.4f}, {r['total_wall_s']:.0f}s{'' if r['complete'] else ' (reset early)'}</li>"
                   for r in st["finished_runs"]) or "<li>none yet</li>"
    pct = 100 * st["version"] / st["budget"]
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="3">
<title>FedCloud aggregation server</title><style>
body{{font:15px system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#1d2330;background:#fff}}
h1{{font-size:22px;margin:0 0 4px}} .sub{{color:#5a6372;margin-bottom:18px}}
.bar{{height:10px;background:#e6e9ef;border-radius:5px;overflow:hidden}} .bar>div{{height:100%;background:#3b6fd8}}
.kpis{{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}} .kpi{{flex:1 1 150px;border:1px solid #e1e4ea;border-radius:8px;padding:10px 12px}}
.kpi b{{display:block;font-size:22px}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
td,th{{text-align:right;padding:4px 8px;border-bottom:1px solid #eef0f4}} th{{color:#5a6372;font-weight:600}}
.wrap{{overflow-x:auto}} code{{background:#f2f4f7;padding:1px 5px;border-radius:4px}}
</style></head><body>
<h1>FedCloud aggregation server (FedAsync)</h1>
<div class="sub">Hosted on Hugging Face Spaces (Docker container, public HTTPS endpoint). Clients on Colab / Kaggle
push model updates only — raw transactions never reach this server.</div>
<div class="bar"><div style="width:{pct:.1f}%"></div></div>
<div class="kpis">
<div class="kpi">Run<b>#{st['run']}</b></div>
<div class="kpi">Updates<b>{st['version']} / {st['budget']}</b></div>
<div class="kpi">AUPRC<b>{st['last_eval'].get('auprc', 0):.4f}</b></div>
<div class="kpi">Traffic<b>{st['last_eval'].get('bytes_total', 0) / 1e6:.2f} MB</b></div>
<div class="kpi">Mean staleness<b>{st['mean_staleness']}</b></div>
</div>
<p><b>Participation:</b> {html.escape(part)}</p>
<div class="wrap"><table><tr><th>~round</th><th>updates</th><th>time (s)</th><th>MB</th><th>AUPRC</th><th>F1</th><th>ROC-AUC</th></tr>
{rows or '<tr><td colspan=7 style="text-align:center">waiting for the first client to pull the model…</td></tr>'}</table></div>
<h3>Finished runs</h3><ul>{done}</ul>
<p style="color:#5a6372">API: <code>/model</code> <code>/update</code> <code>/status</code> <code>/results</code>
· <a href="reset">reset current run</a></p>
</body></html>""".encode()


Base = make_handler(CUR)


class App(Base):
    def do_GET(self):
        p = urlparse(self.path).path.rstrip("/") or "/"
        if p in ("/health", "/model"):
            return super().do_GET()
        if p == "/":
            return self._send(200, page(), {"Content-Type": "text/html; charset=utf-8"})
        if p == "/status":
            return self._send(200, json.dumps(status()).encode(), {"Content-Type": "application/json"})
        if p == "/results":
            return self._send(200, json.dumps([r["run"] for r in RUNS]).encode(), {"Content-Type": "application/json"})
        if p.startswith("/results/"):
            name = p.rsplit("/", 1)[1].removesuffix(".json")
            if name.isdigit() and int(name) < len(RUNS):
                return self._send(200, json.dumps(RUNS[int(name)], indent=1).encode(),
                                  {"Content-Type": "application/json"})
        if p == "/reset":
            CUR.s.done.set()
            return self._send(303, b"", {"Location": "./"})
        return self._send(404, b"not found")


if __name__ == "__main__":
    threading.Thread(target=supervise, daemon=True).start()
    print(f"[hf-space] FedAsync server on :{PORT}; budget={CFG.rounds * CFG.clients} updates per run "
          f"({CFG.rounds} rounds x {CFG.clients} clients)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), App).serve_forever()
