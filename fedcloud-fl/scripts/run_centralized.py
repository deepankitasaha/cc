"""Centralised (non-federated) upper-bound baseline: same MLP, all training data pooled.

    python scripts/run_centralized.py --parts data/parts_iid --epochs 30 --seed 0
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedcloud.data import load_npz  # noqa: E402
from fedcloud.metrics import evaluate  # noqa: E402
from fedcloud.model import init_params, train_local  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="data/parts_iid")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.parts, "client_*.npz")))
    Xs, ys = zip(*[(d.X, d.y) for d in map(load_npz, files)])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    test = load_npz(os.path.join(a.parts, "server_test.npz"))
    w = init_params(a.seed)
    log = {"mode": "centralized", "epochs": a.epochs, "seed": a.seed, "history": []}
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        w = train_local(w, X, y, epochs=1, lr=a.lr, seed=a.seed * 1000 + ep)
        m = evaluate(w, test.X, test.y)
        log["history"].append({"round": ep, "wall_s": round(time.time() - t0, 3), **m})
        print(f"[central] epoch {ep:3d} AUPRC={m['auprc']:.4f} F1={m['f1']:.4f} ROC-AUC={m['roc_auc']:.4f}")
    out = a.out or f"results/centralized_s{a.seed}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(log, open(out, "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
