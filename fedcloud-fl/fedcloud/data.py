"""Data loading, preprocessing, partitioning (IID / Dirichlet label skew) and heterogeneity metrics.

Dataset: ULB Credit Card Fraud (Kaggle "mlg-ulb/creditcardfraud"), 284,807 transactions,
492 frauds (0.172%). Features V1..V28 are PCA components; Time and Amount are raw.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
LABEL = "Class"


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def load_and_preprocess(csv_path: str, test_size: float = 0.2, seed: int = 42):
    """Return (train Split, test Split, scaler stats).

    The test set is a stratified hold-out kept at the aggregation server only.
    Standardisation statistics are computed on the training portion. In a strict FL
    deployment each client would standardise locally; because V1..V28 are already PCA
    outputs, using shared public scaling constants is a common, low-risk simplification.
    """
    df = pd.read_csv(csv_path)
    df["Amount"] = np.log1p(df["Amount"])  # heavy-tailed -> log scale
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[LABEL].to_numpy(dtype=np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    return Split(Xtr, ytr), Split(Xte, yte), {"mean": mu.tolist(), "std": sd.tolist()}


def partition_iid(y: np.ndarray, n_clients: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle and split into equal-sized shards (class ratio preserved in expectation)."""
    idx = rng.permutation(len(y))
    return [np.sort(s) for s in np.array_split(idx, n_clients)]


def partition_dirichlet(y: np.ndarray, n_clients: int, alpha: float, rng: np.random.Generator,
                        min_size: int = 500) -> list[np.ndarray]:
    """Label-distribution skew via Dirichlet(alpha) per class (Hsu et al., 2019).

    For each class c, a proportion vector p_c ~ Dir(alpha * 1_K) decides what share of
    class-c samples each client receives. Small alpha => highly skewed (some clients get
    nearly all frauds, others none). Re-sample until every client has >= min_size samples.
    """
    classes = np.unique(y)
    for _ in range(1000):
        parts: list[list[int]] = [[] for _ in range(n_clients)]
        for c in classes:
            idx_c = rng.permutation(np.where(y == c)[0])
            p = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx_c, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.sort(np.array(p, dtype=np.int64)) for p in parts]
    raise RuntimeError("Could not satisfy min_size; increase alpha or lower min_size")


def make_partitions(y: np.ndarray, n_clients: int, scheme: str, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    if scheme == "iid":
        return partition_iid(y, n_clients, rng)
    if scheme.startswith("dir"):
        return partition_dirichlet(y, n_clients, float(scheme[3:]), rng)
    raise ValueError(f"unknown partition scheme {scheme!r} (use 'iid' or 'dir<alpha>', e.g. dir0.1)")


# ---------------------------------------------------------------- heterogeneity metrics
def _label_dist(y: np.ndarray) -> np.ndarray:
    p1 = float(np.mean(y)) if len(y) else 0.0
    return np.array([1 - p1, p1])


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def heterogeneity_report(y: np.ndarray, parts: list[np.ndarray]) -> dict:
    """Per-client stats + label-skew metrics (cf. Jimenez G. et al., 2024 survey)."""
    g = _label_dist(y)
    clients = []
    for k, idx in enumerate(parts):
        yk = y[idx]
        pk = _label_dist(yk)
        clients.append({
            "client": k, "n": int(len(yk)), "n_fraud": int(yk.sum()),
            "fraud_rate_pct": round(100 * float(pk[1]), 4),
            "hellinger": round(hellinger(pk, g), 5), "jsd": round(jensen_shannon(pk, g), 6),
        })
    sizes = np.array([c["n"] for c in clients], dtype=float)
    w = sizes / sizes.sum()
    return {
        "clients": clients,
        "mean_hellinger": float(np.dot(w, [c["hellinger"] for c in clients])),
        "mean_jsd": float(np.dot(w, [c["jsd"] for c in clients])),
        "frauds_cv": float(np.std([c["n_fraud"] for c in clients]) / max(1e-9, np.mean([c["n_fraud"] for c in clients]))),
    }


# ---------------------------------------------------------------- on-disk partitions
def save_partitions(out_dir: str, train: Split, test: Split, parts: list[np.ndarray], report: dict) -> None:
    """Write one .npz per client (ship ONLY client_k.npz to node k) + server test set."""
    os.makedirs(out_dir, exist_ok=True)
    for k, idx in enumerate(parts):
        np.savez_compressed(os.path.join(out_dir, f"client_{k}.npz"), X=train.X[idx], y=train.y[idx])
    np.savez_compressed(os.path.join(out_dir, "server_test.npz"), X=test.X, y=test.y)
    with open(os.path.join(out_dir, "partition_report.json"), "w") as f:
        json.dump(report, f, indent=2)


def load_npz(path: str) -> Split:
    d = np.load(path)
    return Split(d["X"].astype(np.float32), d["y"].astype(np.float32))
