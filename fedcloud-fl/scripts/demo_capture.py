"""Run one experiment and capture every process's stdout with timestamps (used to build the demo video).

    python scripts/demo_capture.py --mode sync --scheme dir0.5 --rounds 12 --out results/demo_sync_capture.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sync", "async"], default="sync")
    ap.add_argument("--scheme", default="dir0.5")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parts = os.path.join(ROOT, "data", f"parts_{a.scheme}")
    py = sys.executable
    res = os.path.join(ROOT, "results", f"demo_{a.mode}_{a.scheme}.json")
    if a.mode == "sync":
        srv = [py, "-m", "fedcloud.sync_server", "--address", f"127.0.0.1:{a.port}", "--rounds", str(a.rounds),
               "--test", f"{parts}/server_test.npz", "--out", res]
        cli = lambda k: [py, "-m", "fedcloud.sync_client", "--server", f"127.0.0.1:{a.port}", "--cid", str(k),
                         "--data", f"{parts}/client_{k}.npz"]
    else:
        srv = [py, "-m", "fedcloud.async_server", "--host", "127.0.0.1", "--port", str(a.port + 1),
               "--rounds", str(a.rounds), "--test", f"{parts}/server_test.npz", "--out", res]
        cli = lambda k: [py, "-m", "fedcloud.async_client", "--server", f"http://127.0.0.1:{a.port + 1}",
                         "--cid", str(k), "--data", f"{parts}/client_{k}.npz"]
    lines, lock, t0 = [], threading.Lock(), time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    def pump(name, proc):
        for raw in proc.stdout:
            with lock:
                lines.append({"t": round(time.time() - t0, 3), "src": name, "line": ANSI.sub("", raw.rstrip())})

    procs = [("server", subprocess.Popen(srv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))]
    lines.append({"t": 0.0, "src": "server", "line": "$ " + " ".join(srv[1:]).replace(ROOT + "/", "")})
    time.sleep(2.5)
    for k in range(5):
        c = cli(k)
        lines.append({"t": round(time.time() - t0, 3), "src": f"client{k}",
                      "line": "$ " + " ".join(c[1:]).replace(ROOT + "/", "")})
        procs.append((f"client{k}", subprocess.Popen(c, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                                     stderr=subprocess.STDOUT, text=True)))
    ths = [threading.Thread(target=pump, args=p) for p in procs]
    [t.start() for t in ths]
    [p.wait() for _, p in procs]
    [t.join() for t in ths]
    json.dump({"mode": a.mode, "scheme": a.scheme, "lines": sorted(lines, key=lambda x: x["t"])},
              open(a.out, "w"), indent=0)
    print("captured", len(lines), "lines ->", a.out)


if __name__ == "__main__":
    main()
