"""Flower client node for synchronous FedAvg / FedProx.

Run on each client node (Colab / Kaggle / VM), with ONLY that node's partition file:
    python -m fedcloud.sync_client --server <SERVER_IP>:8080 --cid 0 --data data/parts/client_0.npz
"""
from __future__ import annotations

import argparse
import time

import flwr as fl

from .data import load_npz
from .model import train_local
from .netemu import NetEmu
from .serialization import payload_bytes


class FraudClient(fl.client.NumPyClient):
    def __init__(self, cid: int, data, emu: NetEmu):
        self.cid, self.data, self.emu = cid, data, emu

    def fit(self, parameters, config):
        print(f"[client {self.cid}] round {config.get('round')} received from server: "
              f"epochs={config.get('epochs')} lr={config.get('lr')} prox_mu={config.get('prox_mu')} "
              f"seed={config.get('seed')}", flush=True)
        emu_s = self.emu.transfer(payload_bytes(parameters))  # emulated download
        t0 = time.time()
        epochs = int(config.get("epochs", 1))
        new = train_local(parameters, self.data.X, self.data.y, epochs=epochs, lr=float(config.get("lr", 1e-3)),
                          prox_mu=float(config.get("prox_mu", 0.0)),
                          seed=int(config.get("seed", 0)) * 1000 + int(config.get("round", 0)) * 10 + self.cid)
        train_s = time.time() - t0
        emu_s += self.emu.pad_compute(t0, epochs)
        emu_s += self.emu.transfer(payload_bytes(new))  # emulated upload
        print(f"[client {self.cid}] round {config.get('round')} trained on {len(self.data)} rows "
              f"in {train_s:.2f}s (+{emu_s:.2f}s emulated)", flush=True)
        return new, len(self.data), {"cid": self.cid, "train_s": round(train_s, 4), "emu_s": round(emu_s, 4)}

    def evaluate(self, parameters, config):  # global evaluation happens on the server
        return 0.0, len(self.data), {}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="127.0.0.1:8080")
    ap.add_argument("--cid", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--profiles", default="configs/profiles.json")
    ap.add_argument("--profile-offset", type=int, default=0, help="client k uses profile (k+offset) mod P")
    ap.add_argument("--no-emu", action="store_true", help="disable WAN/compute emulation (real cloud runs)")
    a = ap.parse_args(argv)
    emu = NetEmu(None) if a.no_emu else NetEmu.from_file(a.profiles, a.cid, a.profile_offset)
    data = load_npz(a.data)
    n_fraud = int(data.y.sum())
    print(f"[client {a.cid}] local data {a.data}: {len(data)} rows, {n_fraud} frauds "
          f"({100 * n_fraud / max(1, len(data)):.3f}% fraud) - this data never leaves this window", flush=True)
    if emu.enabled:
        print(f"[client {a.cid}] emulated profile: {emu.p['name']} rtt={emu.p['rtt_ms']}ms "
              f"bw={emu.p['bw_mbps']}Mbit/s compute x{emu.p['compute_factor']}", flush=True)
    client = FraudClient(a.cid, data, emu)
    for attempt in range(30):  # server may not be up yet
        try:
            fl.client.start_client(server_address=a.server, client=client.to_client())
            return
        except Exception as e:  # noqa: BLE001
            print(f"[client {a.cid}] connect failed ({e.__class__.__name__}); retry {attempt+1}/30", flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
