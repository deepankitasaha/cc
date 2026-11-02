"""Asynchronous aggregation server implementing FedAsync (Xie, Koyejo & Gupta, 2019).

Flower's built-in strategies are round-based (synchronous), so the asynchronous variant is a
small HTTP service using the SAME model, data, serialisation and WAN emulator as the Flower
path, so the two are directly comparable.

Protocol
  GET  /model                      -> packed global weights, header X-Version: t
  POST /update?cid=k&base=v&n=n_k  -> body: packed client weights trained from version v
Server update on arrival (no waiting for other clients):
  tau     = t - v                               (staleness)
  alpha_t = alpha * (1 + tau) ** (-a)           (polynomial staleness discount)
  w      <- (1 - alpha_t) * w + alpha_t * w_k ;  t <- t + 1
Global evaluation runs every K updates (K = #clients, i.e. one "round-equivalent").
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .data import load_npz
from .metrics import evaluate
from .model import init_params, n_params
from .serialization import pack, unpack


class AsyncState:
    def __init__(self, a, test):
        self.a, self.test = a, test
        self.w = init_params(a.seed)
        self.version = 0
        self.lock = threading.Lock()
        self.budget = a.rounds * a.clients
        self.bytes_down = self.bytes_up = 0
        self.t0 = None
        self.done = threading.Event()
        self.participation = Counter()
        self.staleness = []
        self.log = {"mode": "async", "strategy": "fedasync", "rounds_cfg": a.rounds, "clients": a.clients,
                    "epochs": a.epochs, "lr": a.lr, "alpha": a.alpha, "staleness_a": a.staleness_a,
                    "seed": a.seed, "tag": a.tag, "n_params": n_params(self.w), "history": [], "updates": []}

    def record_eval(self, w, version):
        m = evaluate(w, self.test.X, self.test.y)
        entry = {"round": version // self.a.clients, "updates": version,
                 "wall_s": round(time.time() - self.t0, 3), "bytes_total": self.bytes_down + self.bytes_up,
                 "bytes_down": self.bytes_down, "bytes_up": self.bytes_up, **m}
        self.log["history"].append(entry)
        print(f"[async-server] upd {version:4d} (~round {entry['round']:3d})  t={entry['wall_s']:7.1f}s  "
              f"MB={entry['bytes_total']/1e6:7.3f}  AUPRC={m['auprc']:.4f}  F1={m['f1']:.4f}", flush=True)


def make_handler(S: AsyncState):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence default access log
            pass

        def _send(self, code, body=b"", headers=None):
            self.send_response(code)
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/health"):
                return self._send(200, b"ok")
            if S.done.is_set():
                return self._send(410, b"done")
            with S.lock:
                if S.t0 is None:  # clock starts at first model pull
                    S.t0 = time.time()
                    S.record_eval(S.w, 0)
                body, ver = pack(S.w), S.version
                S.bytes_down += len(body)
            self._send(200, body, {"X-Version": ver, "X-Epochs": S.a.epochs, "X-Lr": S.a.lr, "X-Seed": S.a.seed})

        def do_POST(self):
            q = parse_qs(urlparse(self.path).query)
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if S.done.is_set():
                return self._send(410, b"done")
            wk = unpack(body)
            cid, base = int(q["cid"][0]), int(q["base"][0])
            snap = None
            with S.lock:
                S.bytes_up += len(body)
                tau = S.version - base
                alpha_t = S.a.alpha * (1.0 + tau) ** (-S.a.staleness_a)
                S.w = [((1 - alpha_t) * w + alpha_t * u).astype("float32") for w, u in zip(S.w, wk)]
                S.version += 1
                S.participation[cid] += 1
                S.staleness.append(tau)
                S.log["updates"].append({"v": S.version, "cid": cid, "tau": tau, "alpha_t": round(alpha_t, 4),
                                         "t": round(time.time() - S.t0, 3)})
                if S.version % S.a.clients == 0 or S.version >= S.budget:
                    snap = ([w.copy() for w in S.w], S.version)
                    S.record_eval(*snap)
                if S.version >= S.budget:
                    S.done.set()
            self._send(200, b"ok", {"X-Version": S.version})

    return H


def finalize_log(S: AsyncState, t_start: float) -> dict:
    """Adds the end-of-run summary (wall time, participation, staleness) to S.log and returns it."""
    S.log["total_wall_s"] = round(time.time() - (S.t0 or t_start), 3)
    S.log["participation"] = {str(k): v for k, v in sorted(S.participation.items())}
    S.log["mean_staleness"] = sum(S.staleness) / max(1, len(S.staleness))
    S.log["max_staleness"] = max(S.staleness or [0])
    return S.log


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--rounds", type=int, default=30, help="budget = rounds * clients client updates")
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.4, help="FedAsync mixing weight")
    ap.add_argument("--staleness-a", type=float, default=0.5, help="polynomial staleness exponent")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test", default="data/parts/server_test.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/async.json")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    S = AsyncState(a, load_npz(a.test))
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(S))
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    print(f"[async-server] listening on {a.host}:{a.port}; budget={S.budget} updates | hyperparameters sent "
          f"with every model: epochs={a.epochs} lr={a.lr} seed={a.seed}; mixing alpha={a.alpha} "
          f"staleness exponent a={a.staleness_a}", flush=True)
    t_start = time.time()
    S.done.wait()
    time.sleep(1.0)  # let in-flight clients receive 410
    httpd.shutdown()
    finalize_log(S, t_start)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(S.log, f, indent=1)
    print(f"[async-server] participation={dict(S.participation)} mean staleness={S.log['mean_staleness']:.2f}")
    print(f"[async-server] wrote {a.out}")


if __name__ == "__main__":
    main()
