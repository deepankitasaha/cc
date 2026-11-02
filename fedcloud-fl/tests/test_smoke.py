"""Fast sanity tests (no dataset or network needed):  python -m pytest tests -q  (or python tests/test_smoke.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedcloud.data import heterogeneity_report, make_partitions  # noqa: E402
from fedcloud.metrics import evaluate  # noqa: E402
from fedcloud.model import _grads, forward, init_params, n_params, train_local  # noqa: E402
from fedcloud.serialization import pack, payload_bytes, unpack  # noqa: E402


def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 30)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 1.8).astype(np.float32)
    return X, y


def test_param_count():
    assert n_params(init_params()) == 30 * 64 + 64 + 64 * 32 + 32 + 32 + 1 == 4097


def test_gradient_matches_finite_difference():
    X, y = _toy(64)
    p = init_params(1)
    g = _grads(p, X, y, 3.0)

    def loss(pp):
        o = np.clip(forward(pp, X), 1e-7, 1 - 1e-7)
        w = np.where(y == 1, 3.0, 1.0)
        return np.mean(-w * (y * np.log(o) + (1 - y) * np.log(1 - o)))
    for li, idx in [(0, (3, 5)), (2, (10, 7)), (4, (4, 0)), (5, (0,))]:
        q = [a.astype(np.float64) for a in p]
        q[li][idx] += 1e-4
        up = loss(q)
        q[li][idx] -= 2e-4
        dn = loss(q)
        assert abs((up - dn) / 2e-4 - g[li][idx]) < 1e-3


def test_training_learns():
    X, y = _toy()
    p = train_local(init_params(0), X, y, epochs=5, pos_weight=3.0, lr=3e-3)
    assert evaluate(p, X, y)["roc_auc"] > 0.95


def test_partitions_cover_all_once():
    y = np.r_[np.zeros(9000), np.ones(100)].astype(np.float32)
    for sc in ["iid", "dir0.5"]:
        parts = make_partitions(y, 5, sc, 0)
        allidx = np.concatenate(parts)
        assert len(allidx) == len(y) and len(np.unique(allidx)) == len(y)
    rep_iid = heterogeneity_report(y, make_partitions(y, 5, "iid", 0))
    rep_dir = heterogeneity_report(y, make_partitions(y, 5, "dir0.1", 0))
    assert rep_dir["frauds_cv"] > rep_iid["frauds_cv"]


def test_serialization_roundtrip():
    p = init_params(2)
    q = unpack(pack(p))
    assert all(np.array_equal(a, b) for a, b in zip(p, q))
    assert payload_bytes(p) > 4097 * 4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
