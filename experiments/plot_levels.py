"""Figure for the three-level experiment, from experiments/results/<tag>/results.csv (no simulation needed).

A: intact performance per environment.  B: performance retained as modules are disabled (mean over environments).
C: performance retained after disabling one module, per environment.
Retained = mean perf with k modules disabled / mean perf intact, ratio of means within each environment
(averaged over all disabled-module subsets of size k); 95% CIs bootstrap over test robots (seed x grouping).
"""
import os, sys
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABEL = {"flat": "Flat (one controller)", "multi": "Multi-level, no communication",
         "multi_comm": "Multi-level + communication"}
COLOR = {"flat": "#7f7f7f", "multi": "#1f77b4", "multi_comm": "#e8710a"}
rng = np.random.default_rng(0)


def retained(df, k, units):
    """Mean over envs of [mean perf(k) / mean perf(0)] for the given robot units."""
    out = []
    for env, d in df.groupby("env"):
        d = d[d["unit"].isin(units)]
        base = d[d.k == 0].groupby("unit").perf.mean().mean()
        dis = d[d.k == k].groupby("unit").perf.mean().mean()
        out.append(dis / base if base > 1e-6 else np.nan)
    return np.nanmean(out)


def boot(df, k, n=500):
    units = df["unit"].unique()
    vals = [retained(df, k, rng.choice(units, len(units))) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def main(tag, out):
    df = pd.read_csv(os.path.join("experiments", "results", tag, "results.csv"))
    df["unit"] = df["seed"].astype(str) + "_" + df["robot"].astype(str)
    conds = [c for c in LABEL if c in df.cond.unique()]
    envs = list(df.env.unique())
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6, "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.1), gridspec_kw={"width_ratios": [1.2, 1, 1.2]})

    # A: intact performance per env
    w = 0.8 / len(conds)
    for i, c in enumerate(conds):
        m = [df[(df.cond == c) & (df.env == e) & (df.k == 0)].groupby("unit").perf.mean() for e in envs]
        ax[0].bar(np.arange(len(envs)) + (i - (len(conds) - 1) / 2) * w, [x.mean() for x in m], w,
                  yerr=[1.96 * x.std() / np.sqrt(len(x)) for x in m], color=COLOR[c], label=LABEL[c], capsize=1, error_kw=dict(lw=0.6))
    ax[0].set_xticks(range(len(envs)), [e.replace("_", "\n") for e in envs])
    ax[0].set_ylabel("Distance travelled")
    ax[0].set_title("A  Intact", loc="left", fontweight="bold")

    # B: retained vs number of disabled modules
    ks = [0, 1, 2, 3, 4]
    for c in conds:
        d = df[df.cond == c]
        mean = [retained(d, k, d.unit.unique()) for k in ks]
        ci = np.array([boot(d, k) if 0 < k < 4 else (mean[k], mean[k]) for k in ks])
        ax[1].plot(ks, mean, "-o", color=COLOR[c], label=LABEL[c], ms=2.5, lw=1)
        ax[1].fill_between(ks, ci[:, 0], ci[:, 1], color=COLOR[c], alpha=0.18, lw=0)
    ax[1].set_xticks(ks, ["0\n(intact)", "1", "2", "3", "4\n(all)"])
    ax[1].set_xlabel("Modules disabled (of 4)")
    ax[1].set_ylabel("Performance retained")
    ax[1].set_ylim(-0.05, 1.1)
    ax[1].set_title("B  Degradation", loc="left", fontweight="bold")
    
    # C: retained after one module disabled, per env
    for i, c in enumerate(conds):
        d = df[df.cond == c]
        vals, errs = [], []
        for e in envs:
            de = d[d.env == e]
            units = de.unit.unique()
            r = retained(de, 1, units)
            b = [retained(de, 1, rng.choice(units, len(units))) for _ in range(300)]
            vals.append(r)
            errs.append(np.percentile(b, [2.5, 97.5]) - r)
        errs = np.abs(np.array(errs).T)
        ax[2].bar(np.arange(len(envs)) + (i - (len(conds) - 1) / 2) * w, vals, w, yerr=errs,
                  color=COLOR[c], capsize=1, error_kw=dict(lw=0.6))
    ax[2].set_xticks(range(len(envs)), [e.replace("_", "\n") for e in envs])
    ax[2].set_ylabel("Performance retained")
    ax[2].set_title("C  One module lost", loc="left", fontweight="bold")
    ax[2].axhline(1, color="k", lw=0.5, ls=":")
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.5, handlelength=1.2, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1), pad=0.4, w_pad=0.8)
    path = os.path.join("experiments", "results", tag, out)
    fig.savefig(path + ".png", dpi=300)
    fig.savefig(path + ".pdf")
    print("saved", path)

    # summary numbers for the text
    print("\nIntact perf (mean over envs) and retained fraction:")
    for c in conds:
        d = df[df.cond == c]
        print(f"{c:11s} intact {d[d.k == 0].perf.mean():.3f} | retained k=1 {retained(d, 1, d.unit.unique()):.2f}"
              f"  k=2 {retained(d, 2, d.unit.unique()):.2f}  k=3 {retained(d, 3, d.unit.unique()):.2f}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--tag", default="main")
    p.add_argument("--out", default="levels_figure")
    a = p.parse_args()
    main(a.tag, a.out)
