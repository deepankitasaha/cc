"""Aggregate results/*.json -> paper figures (PDF) + LaTeX tables + results/summary.csv.

    python scripts/make_figures.py [--results results] [--paper ../paper]

Runs are grouped by name with the seed suffix stripped, e.g. sync_fedavg_dir0.5_E1_s{0,1,2},
and reported as mean ± std. Files named cloud_*.json (real cross-cloud runs, see
deploy/CLOUD_DEPLOYMENT.md) are listed as individual rows.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TARGET = 0.70  # AUPRC target used for rounds/time/MB-to-target
# Reference categorical palette (validated order) + neutral for the centralized baseline
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "violet": "#6250d6", "gray": "#6b6a66"}
SCHEME_STYLE = {"iid": (C["blue"], "-", "o"), "dir1.0": (C["orange"], "-", "s"),
                "dir0.5": (C["aqua"], "-", "^"), "dir0.1": (C["violet"], "-", "D")}
SCHEME_LABEL = {"iid": "IID", "dir1.0": r"Dir($\alpha$=1.0)", "dir0.5": r"Dir($\alpha$=0.5)",
                "dir0.1": r"Dir($\alpha$=0.1)"}
SCHEME_TEX = {"iid": "IID", "dir1.0": r"Dir($\alpha{=}1.0$)", "dir0.5": r"Dir($\alpha{=}0.5$)",
              "dir0.1": r"Dir($\alpha{=}0.1$)"}

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "grid.linewidth": 0.5, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b2a",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.frameon": False,
                     "lines.linewidth": 1.4, "lines.markersize": 3.5, "figure.dpi": 150})


def load(results):
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(results, "*.json"))):
        name = os.path.basename(f)[:-5]
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        if "history" not in d:
            continue
        key = name if name.startswith("cloud_") else re.sub(r"_s\d+", "", name)
        d["_name"] = name
        groups[key].append(d)
    return groups


def first_hit(h, key):
    for e in h:
        if e["auprc"] >= TARGET:
            return e[key]
    return np.nan


def stats(runs):
    def ms(vals):
        v = np.array(vals, dtype=float)
        ok = v[~np.isnan(v)]
        return (float(ok.mean()) if len(ok) else np.nan, float(ok.std()) if len(ok) > 1 else 0.0, len(ok), len(v))

    fin = lambda k: [r["history"][-1][k] for r in runs]
    return {
        "auprc": ms(fin("auprc")), "f1": ms(fin("f1")), "roc_auc": ms(fin("roc_auc")),
        "recall": ms(fin("recall")), "precision": ms(fin("precision")), "accuracy": ms(fin("accuracy")),
        "best_auprc": ms([max(e["auprc"] for e in r["history"]) for r in runs]),
        "wall": ms([r["history"][-1]["wall_s"] for r in runs]),
        "mb": ms([r["history"][-1].get("bytes_total", 0) / 1e6 for r in runs]),
        "rt_round": ms([first_hit(r["history"], "round") for r in runs]),
        "rt_wall": ms([first_hit(r["history"], "wall_s") for r in runs]),
        "rt_mb": ms([first_hit(r["history"], "bytes_total") / 1e6 if "bytes_total" in r["history"][0] else np.nan
                     for r in runs]),
    }


def curve(runs, x, y="auprc"):
    n = min(len(r["history"]) for r in runs)
    X = np.array([[r["history"][i][x] for i in range(n)] for r in runs], dtype=float)
    Y = np.array([[r["history"][i][y] for i in range(n)] for r in runs], dtype=float)
    return X.mean(0), Y.mean(0), Y.std(0)


def fmt(m, d=3, pct=False):
    mean, sd, k, n = m
    if np.isnan(mean):
        return "--"
    s = f"{mean:.{d}f}" + (rf"$\pm${sd:.{d}f}" if n > 1 else "")
    if k < n:
        s += rf"$^{{{k}/{n}}}$"
    return s


def plot_band(ax, x, y, s, color, ls="-", marker=None, label=None, every=3):
    ax.plot(x, y, color=color, ls=ls, marker=marker, markevery=every, label=label)
    ax.fill_between(x, np.clip(y - s, 0, 1), np.clip(y + s, 0, 1), color=color, alpha=0.12, lw=0)


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--results", default=os.path.join(root, "results"))
    ap.add_argument("--paper", default=os.path.join(os.path.dirname(root), "paper"))
    a = ap.parse_args()
    figd, tabd = os.path.join(a.paper, "figures"), os.path.join(a.paper, "tables")
    os.makedirs(figd, exist_ok=True)
    os.makedirs(tabd, exist_ok=True)
    G = load(a.results)
    S = {k: stats(v) for k, v in G.items()}
    cen = [k for k in G if k.startswith("centralized")]
    central = G.get("centralized", [])

    # ---------------- summary.csv
    with open(os.path.join(a.results, "summary.csv"), "w") as f:
        cols = ["auprc", "f1", "roc_auc", "recall", "precision", "accuracy", "wall", "mb", "rt_round", "rt_wall", "rt_mb"]
        f.write("config,n_seeds," + ",".join(f"{c}_mean,{c}_std" for c in cols) + "\n")
        for k in sorted(S):
            f.write(f"{k},{len(G[k])}," + ",".join(f"{S[k][c][0]:.5f},{S[k][c][1]:.5f}" for c in cols) + "\n")

    # ---------------- Table: partitions
    rows = []
    for sc in ["iid", "dir1.0", "dir0.5", "dir0.1"]:
        p = os.path.join(root, "data", f"parts_{sc}", "partition_report.json")
        if os.path.exists(p):
            r = json.load(open(p))
            sizes = "/\\allowbreak ".join(f"{c['n']/1000:.1f}k" for c in r["clients"])
            frauds = "/\\allowbreak ".join(str(c["n_fraud"]) for c in r["clients"])
            rows.append(rf"{SCHEME_TEX[sc]} & {sizes} & {frauds} & {r['mean_hellinger']:.4f} & {r['frauds_cv']:.2f} \\")
    with open(os.path.join(tabd, "partitions.tex"), "w") as f:
        f.write("\\begin{tabular}{lp{2.45cm}p{1.75cm}cc}\n\\toprule\nPartition & Client sizes & Frauds per client & $\\bar{H}$ & CV$_\\text{fraud}$ \\\\\n\\midrule\n")
        f.write("\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- Table: main results
    order = []
    if "centralized" in S:
        order.append(("centralized", "Centralized", "--"))
    for sc in ["iid", "dir1.0", "dir0.5", "dir0.1"]:
        for k, lab in [(f"sync_fedavg_{sc}_E1", "FedAvg (sync)"), (f"sync_fedprox_{sc}_E1", "FedProx (sync)"),
                       (f"async_fedasync_{sc}_E1", "FedAsync"), (f"async_fedasync_{sc}_E1_po4", "FedAsync$^\\dagger$")]:
            if k in S:
                order.append((k, lab, SCHEME_TEX[sc]))
    for k in sorted(G):
        if k.startswith("cloud_"):
            order.append((k, "\\texttt{" + k.replace("_", "\\_") + "}", "real cloud"))
    with open(os.path.join(tabd, "main_results.tex"), "w") as f:
        f.write("\\begin{tabular}{llcccccc}\n\\toprule\nData & Method & AUPRC & F1 & ROC-AUC & Recall & Time (s) & MB \\\\\n\\midrule\n")
        prev = None
        for k, lab, sc in order:
            if prev is not None and sc != prev:
                f.write("\\midrule\n")
            prev = sc
            s = S[k]
            wall = fmt(s["wall"], 1) if k != "centralized" else "--"
            mb = fmt(s["mb"], 2) if k != "centralized" else "0"
            f.write(f"{sc} & {lab} & {fmt(s['auprc'])} & {fmt(s['f1'])} & {fmt(s['roc_auc'])} & {fmt(s['recall'])} & {wall} & {mb} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---------------- Table: sync vs async efficiency to target
    with open(os.path.join(tabd, "sync_async.tex"), "w") as f:
        f.write("\\begin{tabular}{llcccc}\n\\toprule\nData & Method & Rounds$\\to$T & Time$\\to$T (s) & MB$\\to$T & MB/round \\\\\n\\midrule\n")
        for sc in ["iid", "dir1.0", "dir0.5", "dir0.1"]:
            for k, lab in [(f"sync_fedavg_{sc}_E1", "FedAvg"), (f"async_fedasync_{sc}_E1", "FedAsync"),
                           (f"async_fedasync_{sc}_E1_po4", "FedAsync$^\\dagger$")]:
                if k not in S:
                    continue
                s = S[k]
                mbr = s["mb"][0] / G[k][0]["history"][-1]["round"]
                f.write(f"{SCHEME_TEX[sc]} & {lab} & {fmt(s['rt_round'], 1)} & {fmt(s['rt_wall'], 1)} & {fmt(s['rt_mb'], 2)} & {mbr:.3f} \\\\\n")
            if sc != "dir0.1":
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---------------- Table: local epochs
    with open(os.path.join(tabd, "epochs.tex"), "w") as f:
        f.write("\\begin{tabular}{ccccc}\n\\toprule\n$E$ & Final AUPRC & Rounds$\\to$T & MB$\\to$T & Time (s) \\\\\n\\midrule\n")
        for E in [1, 2, 5]:
            k = f"sync_fedavg_dir0.5_E{E}"
            if k in S:
                s = S[k]
                f.write(f"{E} & {fmt(s['auprc'])} & {fmt(s['rt_round'], 1)} & {fmt(s['rt_mb'], 2)} & {fmt(s['wall'], 1)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---------------- Fig 1: FedAvg AUPRC vs round per partition (+ centralized)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for sc in ["iid", "dir1.0", "dir0.5", "dir0.1"]:
        k = f"sync_fedavg_{sc}_E1"
        if k in G:
            col, ls, mk = SCHEME_STYLE[sc]
            x, y, s = curve(G[k], "round")
            plot_band(axs[0], x, y, s, col, ls, mk, SCHEME_LABEL[sc])
            x, y, s = curve(G[k], "round", "roc_auc")
            plot_band(axs[1], x, y, s, col, ls, mk, SCHEME_LABEL[sc])
    if central:
        for ax, met in zip(axs, ["auprc", "roc_auc"]):
            x, y, s = curve(central, "round", met)
            ax.plot(x, y, color=C["gray"], ls="--", label="Centralized")
    axs[0].set(xlabel="Communication round", ylabel="AUPRC (test)", ylim=(0, 0.9))
    axs[1].set(xlabel="Communication round", ylabel="ROC-AUC (test)", ylim=(0, 1.0))
    axs[0].legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(figd, "fedavg_noniid.pdf"))
    fig.savefig(os.path.join(figd, "fedavg_noniid.png"), dpi=200)
    plt.close(fig)

    # ---------------- Fig 2: sync vs async, AUPRC vs wall-clock and vs MB
    scs = [s for s in ["iid", "dir0.5", "dir0.1"] if f"sync_fedavg_{s}_E1" in G]
    fig, axs = plt.subplots(2, len(scs), figsize=(7.0, 3.9), squeeze=False)
    for j, sc in enumerate(scs):
        for k, lab, col, mk in [(f"sync_fedavg_{sc}_E1", "FedAvg (sync)", C["blue"], "o"),
                                (f"sync_fedprox_{sc}_E1", "FedProx (sync)", C["aqua"], "^"),
                                (f"async_fedasync_{sc}_E1", "FedAsync", C["orange"], "s"),
                                (f"async_fedasync_{sc}_E1_po4", "FedAsync, fraud-rich node slow", C["magenta"], "D")]:
            if k not in G:
                continue
            x, y, s = curve(G[k], "wall_s")
            plot_band(axs[0, j], x, y, s, col, "-", mk, lab, every=5)
            x, y, s = curve(G[k], "bytes_total")
            plot_band(axs[1, j], x / 1e6, y, s, col, "-", mk, lab, every=5)
        axs[0, j].set_title(SCHEME_LABEL[sc], fontsize=8)
        axs[0, j].set(xlabel="Wall-clock time (s)", ylim=(0, 0.9))
        axs[1, j].set(xlabel="Cumulative traffic (MB)", ylim=(0, 0.9))
        for i in range(2):
            axs[i, j].axhline(TARGET, color="#b9b8b2", lw=0.7, ls=":")
    axs[0, 0].set_ylabel("AUPRC")
    axs[1, 0].set_ylabel("AUPRC")
    h, l = axs[0, -1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(figd, "sync_vs_async.pdf"))
    fig.savefig(os.path.join(figd, "sync_vs_async.png"), dpi=200)
    plt.close(fig)

    # ---------------- Fig 3: local epochs trade-off (AUPRC vs MB)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for E, col, mk in [(1, C["blue"], "o"), (2, C["orange"], "s"), (5, C["aqua"], "^")]:
        k = f"sync_fedavg_dir0.5_E{E}"
        if k in G:
            x, y, s = curve(G[k], "bytes_total")
            plot_band(ax, x / 1e6, y, s, col, "-", mk, f"E={E}", every=4)
    ax.axhline(TARGET, color="#b9b8b2", lw=0.7, ls=":")
    ax.set(xlabel="Cumulative traffic (MB)", ylabel="AUPRC", ylim=(0, 0.9))
    ax.legend(loc="lower right", title=r"Local epochs, Dir($\alpha$=0.5)", fontsize=6.5, title_fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(figd, "epochs_tradeoff.pdf"))
    fig.savefig(os.path.join(figd, "epochs_tradeoff.png"), dpi=200)
    plt.close(fig)

    # ---------------- Fig 4: partitions (frauds per client) + async participation
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
    width = 0.2
    for i, sc in enumerate(["iid", "dir1.0", "dir0.5", "dir0.1"]):
        p = os.path.join(root, "data", f"parts_{sc}", "partition_report.json")
        if os.path.exists(p):
            r = json.load(open(p))
            fr = [c["n_fraud"] for c in r["clients"]]
            axs[0].bar(np.arange(len(fr)) + (i - 1.5) * width, fr, width * 0.9, color=SCHEME_STYLE[sc][0],
                       label=SCHEME_LABEL[sc])
    axs[0].set(xlabel="Client", ylabel="# fraud samples", xticks=range(5))
    axs[0].legend(fontsize=6.5)
    for i, (k, lab, col) in enumerate([("async_fedasync_dir0.1_E1", "default profiles", C["orange"]),
                                       ("async_fedasync_dir0.1_E1_po4", "fraud-rich node slow", C["magenta"])]):
        if k in G:
            P = np.array([[r["participation"].get(str(c), 0) for c in range(5)] for r in G[k]], float)
            axs[1].bar(np.arange(5) + (i - 0.5) * 0.35, P.mean(0), 0.33, yerr=P.std(0), color=col, label=lab,
                       error_kw={"lw": 0.6, "capsize": 1.5})
    axs[1].set(xlabel="Client", ylabel="# async updates applied", xticks=range(5), ylim=(0, 68))
    axs[1].legend(fontsize=6.5, title=r"FedAsync, Dir($\alpha$=0.1)", title_fontsize=6.5, ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(figd, "partitions_participation.pdf"))
    fig.savefig(os.path.join(figd, "partitions_participation.png"), dpi=200)
    plt.close(fig)

    # ---------------- macros with headline numbers for the text
    def m(k, f, d=3):
        return f"{S[k][f][0]:.{d}f}" if k in S and not np.isnan(S[k][f][0]) else "--"
    macros = {
        "CenAUPRC": m("centralized", "auprc"), "CenFone": m("centralized", "f1"), "CenAUC": m("centralized", "roc_auc"),
        "CenAcc": m("centralized", "accuracy", 4),
        "IIDAvgAUPRC": m("sync_fedavg_iid_E1", "auprc"), "IIDAsyncAUPRC": m("async_fedasync_iid_E1", "auprc"),
        "DirOneAvgAUPRC": m("sync_fedavg_dir1.0_E1", "auprc"), "DirHalfAvgAUPRC": m("sync_fedavg_dir0.5_E1", "auprc"),
        "DirTenthAvgAUPRC": m("sync_fedavg_dir0.1_E1", "auprc"), "DirTenthAvgAUC": m("sync_fedavg_dir0.1_E1", "roc_auc"),
        "DirTenthAvgAcc": m("sync_fedavg_dir0.1_E1", "accuracy", 4),
        "DirTenthProxAUPRC": m("sync_fedprox_dir0.1_E1", "auprc"), "DirHalfProxAUPRC": m("sync_fedprox_dir0.5_E1", "auprc"),
        "DirTenthAsyncAUPRC": m("async_fedasync_dir0.1_E1", "auprc"),
        "DirTenthAsyncSlowAUPRC": m("async_fedasync_dir0.1_E1_po4", "auprc"),
        "DirHalfAsyncAUPRC": m("async_fedasync_dir0.5_E1", "auprc"), "DirOneAsyncAUPRC": m("async_fedasync_dir1.0_E1", "auprc"),
        "IIDAvgWall": m("sync_fedavg_iid_E1", "wall", 1), "IIDAsyncWall": m("async_fedasync_iid_E1", "wall", 1),
        "IIDAvgMB": m("sync_fedavg_iid_E1", "mb", 2), "IIDAsyncMB": m("async_fedasync_iid_E1", "mb", 2),
        "IIDAvgRtWall": m("sync_fedavg_iid_E1", "rt_wall", 1), "IIDAsyncRtWall": m("async_fedasync_iid_E1", "rt_wall", 1),
        "IIDAvgRtMB": m("sync_fedavg_iid_E1", "rt_mb", 2), "IIDAsyncRtMB": m("async_fedasync_iid_E1", "rt_mb", 2),
        "DirHalfAvgRtMB": m("sync_fedavg_dir0.5_E1", "rt_mb", 2), "DirHalfEFiveRtMB": m("sync_fedavg_dir0.5_E5", "rt_mb", 2),
        "DirHalfEFiveAUPRC": m("sync_fedavg_dir0.5_E5", "auprc"), "DirHalfETwoAUPRC": m("sync_fedavg_dir0.5_E2", "auprc"),
        "DirHalfEFiveWall": m("sync_fedavg_dir0.5_E5", "wall", 1), "DirHalfAvgWall": m("sync_fedavg_dir0.5_E1", "wall", 1),
    }
    def ratio(k1, k2, f):
        try:
            return f"{S[k1][f][0] / S[k2][f][0]:.1f}"
        except (KeyError, ZeroDivisionError):
            return "--"
    asy = [r for k in G if k.startswith("async_fedasync") and not k.endswith("po4") for r in G[k]]
    macros["AsyncMeanStale"] = f"{np.mean([r['mean_staleness'] for r in asy]):.1f}" if asy else "--"
    macros["AsyncMaxStale"] = str(max(r["max_staleness"] for r in asy)) if asy else "--"
    macros["SpeedupWall"] = ratio("sync_fedavg_iid_E1", "async_fedasync_iid_E1", "wall")
    macros["SpeedupRtIID"] = ratio("sync_fedavg_iid_E1", "async_fedasync_iid_E1", "rt_wall")
    macros["SpeedupRtHalf"] = ratio("sync_fedavg_dir0.5_E1", "async_fedasync_dir0.5_E1", "rt_wall")
    macros["MBperRound"] = f"{S['sync_fedavg_iid_E1']['mb'][0] / 30:.3f}" if "sync_fedavg_iid_E1" in S else "--"
    pr = G.get("async_fedasync_dir0.1_E1", [])
    macros["PartFast"] = f"{np.mean([r['participation'].get('0', 0) for r in pr]):.0f}" if pr else "--"
    macros["PartSlow"] = f"{np.mean([r['participation'].get('4', 0) for r in pr]):.0f}" if pr else "--"
    pr = G.get("async_fedasync_dir0.1_E1_po4", [])
    macros["PartFraudSlow"] = f"{np.mean([r['participation'].get('0', 0) for r in pr]):.0f}" if pr else "--"
    macros["DirTenthAvgRecall"] = m("sync_fedavg_dir0.1_E1", "recall")
    macros["DirTenthAsyncF"] = m("async_fedasync_dir0.1_E1", "f1")
    with open(os.path.join(tabd, "numbers.tex"), "w") as f:
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(json.dumps(macros, indent=1))
    print("figures ->", figd, "| tables ->", tabd)


if __name__ == "__main__":
    main()
