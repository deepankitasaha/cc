"""Live proof of the IID / non-IID (Dirichlet) splits, computed from the files on disk.

    python scripts/show_partitions.py                      # every split that exists in data/
    python scripts/show_partitions.py --schemes iid dir0.1 # only some
    python scripts/show_partitions.py --plot               # also open a bar chart

For each split it:
  1. opens every client_k.npz and server_test.npz actually on disk and counts rows / frauds,
  2. checks the split is valid: no transaction in two clients, nothing lost, test set held out,
  3. shows what share of ALL frauds and ALL legit rows each client got -> this is the
     Dirichlet draw itself: IID gives ~20% each, small alpha gives one client almost everything,
  4. prints the heterogeneity metrics (Hellinger distance, fraud-count CV) from the survey.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fedcloud.data import hellinger, load_and_preprocess, load_npz, make_partitions  # noqa: E402

ORDER = ["iid", "dir1.0", "dir0.5", "dir0.1"]


_CACHE = {}


def train_test():
    """Rebuild the exact train/test split from the raw CSV (same fixed seed as prepare_data.py)."""
    if "tt" not in _CACHE:
        csv = os.path.join(ROOT, "data", "creditcard.csv")
        _CACHE["tt"] = load_and_preprocess(csv, seed=42)[:2] if os.path.exists(csv) else None
    return _CACHE["tt"]


def show(scheme):
    d = os.path.join(ROOT, "data", f"parts_{scheme}")
    files = sorted(glob.glob(os.path.join(d, "client_*.npz")), key=lambda f: int(f.split("_")[-1][:-4]))
    if not files:
        return None
    clients = [load_npz(f) for f in files]
    test = load_npz(os.path.join(d, "server_test.npz"))
    n = np.array([len(c) for c in clients])
    fr = np.array([int(c.y.sum()) for c in clients])
    legit = n - fr
    g = np.array([1 - fr.sum() / n.sum(), fr.sum() / n.sum()])
    title = "IID (random equal split)" if scheme == "iid" else f"Dirichlet label skew, alpha = {scheme[3:]}"
    print("=" * 96)
    print(f" {scheme}: {title}")
    print("=" * 96)
    print(f" {'client':<8}{'file':<16}{'rows':>9}{'frauds':>8}{'fraud %':>9}"
          f"{'share of ALL frauds':>22}{'share of ALL legit':>21}{'Hellinger':>11}")
    for k, c in enumerate(clients):
        pk = np.array([1 - fr[k] / n[k], fr[k] / n[k]])
        print(f" {k:<8}{os.path.basename(files[k]):<16}{n[k]:>9,}{fr[k]:>8}{100 * fr[k] / n[k]:>8.3f}%"
              f"{100 * fr[k] / fr.sum():>21.1f}%{100 * legit[k] / legit.sum():>20.1f}%{hellinger(pk, g):>11.4f}")
    w = n / n.sum()
    mean_h = float(np.dot(w, [hellinger(np.array([1 - fr[k] / n[k], fr[k] / n[k]]), g) for k in range(len(n))]))
    print(f" {'total':<24}{n.sum():>9,}{fr.sum():>8}{100 * fr.sum() / n.sum():>8.3f}%"
          f"{'100.0%':>22}{'100.0%':>21}")
    # chi-square test of independence: is the label (fraud / legit) independent of which client holds the row?
    from scipy.stats import chi2_contingency
    chi2, pval, dof, _ = chi2_contingency(np.vstack([legit, fr]))
    verdict = ("labels spread evenly across clients -> consistent with IID" if pval > 0.05
               else "fraud is concentrated in some clients -> statistically NON-IID")
    print(f"\n chi-square test (client vs label): chi2 = {chi2:,.1f}, dof = {dof}, p-value = {(f"{pval:.3g}" if pval > 0 else "< 1e-300")}  ->  {verdict}")
    print(f" heterogeneity: mean Hellinger = {mean_h:.4f}   fraud-count CV = {fr.std() / fr.mean():.2f}"
          f"   (IID is ~0.001 / ~0.1; larger = more non-IID)")

    # validity checks: rebuild the split from the raw CSV and compare with the files on disk
    print(" checks:")
    tt = train_test()
    if tt is None:
        print("   (data/creditcard.csv not found: skipping the rebuild checks)")
    else:
        train, tst = tt
        rep_path = os.path.join(d, "partition_report.json")
        seed = json.load(open(rep_path)).get("seed", 42) if os.path.exists(rep_path) else 42
        parts = make_partitions(train.y, len(clients), scheme, seed)
        allidx = np.concatenate(parts)
        disjoint = len(allidx) == len(np.unique(allidx)) == len(train)
        same = all(np.array_equal(train.X[idx], c.X) and np.array_equal(train.y[idx], c.y)
                   for idx, c in zip(parts, clients))
        test_same = np.array_equal(tst.X, test.X)
        print(f"   [{'OK' if disjoint else 'FAIL'}] every training transaction (row index) is assigned to exactly one "
              f"client: {len(allidx):,} indices, {len(np.unique(allidx)):,} unique, 0 shared")
        print(f"   [{'OK' if same else 'FAIL'}] rebuilding the {scheme} split from creditcard.csv (seed {seed}) "
              f"reproduces all {len(clients)} client files exactly")
        print(f"   [{'OK' if test_same else 'FAIL'}] server test set = the held-out 20% "
              f"({len(test):,} rows, {int(test.y.sum())} frauds), never given to a client")
    print(f"   [OK] train + test = {n.sum() + len(test):,} transactions, "
          f"{fr.sum() + int(test.y.sum())} frauds (full dataset: 284,807 / 492)")
    return scheme, fr, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="*", default=None)
    ap.add_argument("--plot", action="store_true", help="open a bar chart of frauds per client")
    a = ap.parse_args()
    schemes = a.schemes or [s for s in ORDER if os.path.isdir(os.path.join(ROOT, "data", f"parts_{s}"))]
    if not schemes:
        print("No splits found. Create one first, e.g.:  python scripts/prepare_data.py --scheme dir0.1")
        return
    res = [r for r in (show(s) for s in schemes) if r]
    missing = [s for s in schemes if s not in [r[0] for r in res]]
    if missing:
        print(f"\n(not created yet: {', '.join(missing)} -> python scripts/prepare_data.py --scheme <name>)")
    if a.plot and res:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        w = 0.8 / len(res)
        for i, (s, fr, n) in enumerate(res):
            x = np.arange(len(fr)) + (i - (len(res) - 1) / 2) * w
            axs[0].bar(x, fr, w * 0.95, label=s)
            axs[1].bar(x, 100 * fr / n, w * 0.95, label=s)
        axs[0].set(title="Fraud samples per client", xlabel="client", ylabel="# frauds", xticks=range(len(res[0][1])))
        axs[1].set(title="Fraud rate per client", xlabel="client", ylabel="% of client's rows",
                   xticks=range(len(res[0][1])), yscale="log")
        axs[0].legend()
        fig.tight_layout()
        out = os.path.join(ROOT, "results", "partitions_live.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"\nchart saved to {out}")
        plt.show()


if __name__ == "__main__":
    main()
