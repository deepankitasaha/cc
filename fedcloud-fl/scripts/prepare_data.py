"""Download (optional), preprocess and partition the Credit Card Fraud dataset.

    python scripts/prepare_data.py --csv data/creditcard.csv --clients 5 --scheme dir0.1 --out data/parts_dir0.1

Schemes: iid | dir<alpha>  (Dirichlet label skew; smaller alpha = more non-IID)
Outputs: client_<k>.npz (give node k only its own file), server_test.npz, partition_report.json
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedcloud.data import heterogeneity_report, load_and_preprocess, make_partitions, save_partitions  # noqa: E402

MIRROR = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/creditcard.csv")
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--scheme", default="iid")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not os.path.exists(a.csv):
        print(f"{a.csv} not found -> downloading public mirror (or get it from kaggle.com/datasets/mlg-ulb/creditcardfraud)")
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        urllib.request.urlretrieve(MIRROR, a.csv)
    out = a.out or f"data/parts_{a.scheme}"
    train, test, _ = load_and_preprocess(a.csv, seed=42)  # fixed global test split for all experiments
    parts = make_partitions(train.y, a.clients, a.scheme, a.seed)
    rep = heterogeneity_report(train.y, parts)
    rep.update({"scheme": a.scheme, "seed": a.seed, "n_train": len(train), "n_test": len(test),
                "test_frauds": int(test.y.sum())})
    save_partitions(out, train, test, parts, rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "clients"}, indent=1))
    for c in rep["clients"]:
        print(f"  client {c['client']}: n={c['n']:6d}  frauds={c['n_fraud']:4d}  rate={c['fraud_rate_pct']:.3f}%  "
              f"H={c['hellinger']:.4f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
