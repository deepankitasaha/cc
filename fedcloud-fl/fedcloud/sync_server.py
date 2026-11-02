"""Synchronous aggregation server (FedAvg / FedProx) built on Flower.

Run on the aggregation node (cloud VM / Cloud Run / your laptop):
    python -m fedcloud.sync_server --rounds 30 --clients 5 --strategy fedavg \
        --test data/parts/server_test.npz --out results/run.json --address 0.0.0.0:8080
"""
from __future__ import annotations

import argparse
import json
import os
import time

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from .data import load_npz
from .metrics import evaluate
from .model import init_params, n_params


def _tensor_bytes(parameters) -> int:
    return int(sum(len(t) for t in parameters.tensors))


class LoggingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that records wall-clock time, communication volume and global metrics per round.

    FedProx (Li et al., 2020) uses the same server-side weighted averaging; the proximal
    term is applied on the clients, enabled via the `prox_mu` value sent in the fit config.
    """

    def __init__(self, test, log: dict, **kw):
        self.test = test
        self.log = log
        self.t0 = None
        self.bytes_down = 0
        self.bytes_up = 0
        self.round_start = {}
        super().__init__(evaluate_fn=self._evaluate, **kw)

    def _evaluate(self, rnd, ndarrays, _cfg):
        m = evaluate(ndarrays, self.test.X, self.test.y)
        now = time.time()
        entry = {"round": rnd, "wall_s": round(now - self.t0, 3) if self.t0 else 0.0,
                 "bytes_total": self.bytes_down + self.bytes_up,
                 "bytes_down": self.bytes_down, "bytes_up": self.bytes_up, **m}
        self.log["history"].append(entry)
        print(f"[server] round {rnd:3d}  t={entry['wall_s']:7.1f}s  MB={entry['bytes_total']/1e6:7.3f}  "
              f"AUPRC={m['auprc']:.4f}  F1={m['f1']:.4f}  ROC-AUC={m['roc_auc']:.4f}", flush=True)
        return m["loss"], m

    def configure_fit(self, server_round, parameters, client_manager):
        ins = super().configure_fit(server_round, parameters, client_manager)
        self.round_start[server_round] = time.time()
        if self.t0 is None:  # clock starts when round 1 is dispatched (all clients connected)
            self.t0 = self.round_start[server_round]
        self.bytes_down += _tensor_bytes(parameters) * len(ins)
        return ins

    def aggregate_fit(self, server_round, results, failures):
        self.bytes_up += sum(_tensor_bytes(fr.parameters) for _, fr in results)
        self.log["rounds"].append({
            "round": server_round,
            "duration_s": round(time.time() - self.round_start[server_round], 3),
            "clients": [{"cid": int(fr.metrics.get("cid", -1)), "n": fr.num_examples,
                         "train_s": fr.metrics.get("train_s"), "emu_s": fr.metrics.get("emu_s")}
                        for _, fr in results],
            "failures": len(failures),
        })
        return super().aggregate_fit(server_round, results, failures)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default="0.0.0.0:8080")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedavg")
    ap.add_argument("--prox-mu", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=1, help="local epochs per round")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test", default="data/parts/server_test.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/sync.json")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    test = load_npz(a.test)
    params = init_params(a.seed)
    mu = a.prox_mu if a.strategy == "fedprox" else 0.0
    log = {"mode": "sync", "strategy": a.strategy, "rounds_cfg": a.rounds, "clients": a.clients,
           "epochs": a.epochs, "lr": a.lr, "prox_mu": mu, "seed": a.seed, "tag": a.tag,
           "n_params": n_params(params), "history": [], "rounds": []}
    strategy = LoggingFedAvg(
        test, log,
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=a.clients, min_available_clients=a.clients,
        initial_parameters=ndarrays_to_parameters(params),
        on_fit_config_fn=lambda r: {"round": r, "epochs": a.epochs, "lr": a.lr, "prox_mu": mu, "seed": a.seed},
    )
    print(f"[server] strategy={a.strategy} rounds={a.rounds} clients={a.clients} | hyperparameters sent to "
          f"clients each round: epochs={a.epochs} lr={a.lr} prox_mu={mu} seed={a.seed}", flush=True)
    t_start = time.time()
    fl.server.start_server(server_address=a.address, config=fl.server.ServerConfig(num_rounds=a.rounds),
                           strategy=strategy)
    log["total_wall_s"] = round(time.time() - t_start, 3)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(log, f, indent=1)
    print(f"[server] wrote {a.out}")


if __name__ == "__main__":
    main()
