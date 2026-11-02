"""FedAsync client node: pull model -> train locally -> push update, repeat until server says done.

    python -m fedcloud.async_client --server http://<SERVER_IP>:8081 --cid 0 --data data/parts/client_0.npz
"""
from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request

from .data import load_npz
from .model import train_local
from .netemu import NetEmu
from .serialization import pack, unpack


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    ap.add_argument("--cid", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profiles", default="configs/profiles.json")
    ap.add_argument("--profile-offset", type=int, default=0, help="client k uses profile (k+offset) mod P")
    ap.add_argument("--no-emu", action="store_true")
    a = ap.parse_args(argv)
    data = load_npz(a.data)
    emu = NetEmu(None) if a.no_emu else NetEmu.from_file(a.profiles, a.cid, a.profile_offset)
    n_fraud = int(data.y.sum())
    print(f"[async-client {a.cid}] local data {a.data}: {len(data)} rows, {n_fraud} frauds "
          f"({100 * n_fraud / max(1, len(data)):.3f}% fraud) - this data never leaves this window", flush=True)
    it = 0
    for attempt in range(60):  # wait for server
        try:
            urllib.request.urlopen(a.server + "/health", timeout=5).read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 410:
                return
            break
        except Exception:  # noqa: BLE001
            time.sleep(1)
    while True:
        try:
            with urllib.request.urlopen(a.server + "/model", timeout=60) as r:
                blob, base = r.read(), int(r.headers["X-Version"])
                if r.headers.get("X-Epochs"):  # hyperparameters come from the server
                    a.epochs, a.lr = int(r.headers["X-Epochs"]), float(r.headers["X-Lr"])
                    print(f"[async-client {a.cid}] pulled model v{base}; received from server: "
                          f"epochs={a.epochs} lr={a.lr}", flush=True)
        except urllib.error.HTTPError as e:
            if e.code == 410:
                break
            raise
        emu.transfer(len(blob))
        w = unpack(blob)
        t0 = time.time()
        new = train_local(w, data.X, data.y, epochs=a.epochs, lr=a.lr, seed=a.seed * 1000 + it * 10 + a.cid)
        train_s = time.time() - t0
        emu.pad_compute(t0, a.epochs)
        body = pack(new)
        emu.transfer(len(body))
        req = urllib.request.Request(f"{a.server}/update?cid={a.cid}&base={base}&n={len(data)}", data=body,
                                     method="POST", headers={"Content-Type": "application/octet-stream"})
        try:
            urllib.request.urlopen(req, timeout=60).read()
        except urllib.error.HTTPError as e:
            if e.code == 410:
                break
            raise
        it += 1
        print(f"[async-client {a.cid}] update {it} from v{base} (train {train_s:.2f}s)", flush=True)
    print(f"[async-client {a.cid}] finished after {it} updates", flush=True)


if __name__ == "__main__":
    main()
