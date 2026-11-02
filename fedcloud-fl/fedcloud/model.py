"""Dependency-light MLP for binary fraud detection, written in NumPy.

Architecture: 30 -> 64 -> 32 -> 1 (ReLU, ReLU, sigmoid), He initialisation,
class-weighted binary cross-entropy, Adam optimiser, optional FedProx proximal term.
Pure NumPy keeps every node (Colab, Kaggle, a 1-vCPU free-tier VM, a slim Docker image)
installable in seconds and makes the parameter vector easy to serialise and count.
"""
from __future__ import annotations

import numpy as np

LAYERS = (30, 64, 32, 1)


def init_params(seed: int = 0, layers=LAYERS) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    params = []
    for fan_in, fan_out in zip(layers[:-1], layers[1:]):
        params.append((rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(np.float32))
        params.append(np.zeros(fan_out, dtype=np.float32))
    return params


def n_params(params: list[np.ndarray]) -> int:
    return int(sum(p.size for p in params))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def forward(params, X, cache: bool = False):
    acts = [X]
    h = X
    n_layers = len(params) // 2
    for i in range(n_layers):
        W, b = params[2 * i], params[2 * i + 1]
        z = h @ W + b
        h = np.maximum(z, 0) if i < n_layers - 1 else _sigmoid(z)
        acts.append(h)
    out = h.ravel()
    return (out, acts) if cache else out


def predict_proba(params, X, batch: int = 65536) -> np.ndarray:
    return np.concatenate([forward(params, X[i:i + batch]) for i in range(0, len(X), batch)])


def bce_loss(params, X, y, pos_weight: float = 1.0) -> float:
    p = np.clip(predict_proba(params, X), 1e-7, 1 - 1e-7)
    w = np.where(y == 1, pos_weight, 1.0)
    return float(np.mean(-w * (y * np.log(p) + (1 - y) * np.log(1 - p))))


def _grads(params, X, y, pos_weight):
    out, acts = forward(params, X, cache=True)
    n = len(y)
    w = np.where(y == 1, pos_weight, 1.0).astype(np.float32)
    delta = ((out - y) * w / n).reshape(-1, 1).astype(np.float32)  # dL/dz for sigmoid+BCE
    grads = [None] * len(params)
    n_layers = len(params) // 2
    for i in reversed(range(n_layers)):
        grads[2 * i] = acts[i].T @ delta
        grads[2 * i + 1] = delta.sum(0)
        if i > 0:
            delta = (delta @ params[2 * i].T) * (acts[i] > 0)
    return grads


def train_local(params, X, y, epochs: int = 1, batch_size: int = 256, lr: float = 1e-3,
                pos_weight: float = 10.0, prox_mu: float = 0.0, seed: int = 0):
    """Local training with Adam (fresh optimiser state each round, as is standard in FL).

    prox_mu > 0 adds the FedProx term (mu/2)||w - w_global||^2 (Li et al., 2020), which
    limits client drift under non-IID data.
    """
    rng = np.random.default_rng(seed)
    w = [p.copy() for p in params]
    w_global = [p.copy() for p in params]
    m = [np.zeros_like(p) for p in w]
    v = [np.zeros_like(p) for p in w]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for s in range(0, len(y), batch_size):
            bi = order[s:s + batch_size]
            g = _grads(w, X[bi], y[bi], pos_weight)
            t += 1
            for j in range(len(w)):
                gj = g[j] + (prox_mu * (w[j] - w_global[j]) if prox_mu > 0 else 0)
                m[j] = b1 * m[j] + (1 - b1) * gj
                v[j] = b2 * v[j] + (1 - b2) * gj * gj
                mhat = m[j] / (1 - b1 ** t)
                vhat = v[j] / (1 - b2 ** t)
                w[j] = (w[j] - lr * mhat / (np.sqrt(vhat) + eps)).astype(np.float32)
    return w
