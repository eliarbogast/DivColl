"""Analysis + figure for the pilot (v2), from experiments/results/<tag>/results.csv. No simulation needed.

Statistics (as pre-specified in PLAN.md): retention for a seed = mean over environments of
[mean perf with lesion / mean intact perf] (ratio of means over that seed's test robots, and over all subsets of
size k for module lesions). Mean and 95% CI (t-interval) are taken ACROSS SEEDS. Bootstrap over test robots is
printed as a secondary check only.

Outputs (in results/<tag>/): levels_figure.{png,pdf} (main, 6.5x2.0 in), levels_supp_env.{png,pdf}
(k=1 retention per environment), summary.md (all numbers).
"""
import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

LABEL = {"central": "Single central controller", "multilevel": "Multi-level, no communication",
         "multilevel_comm": "Multi-level + communication",
         "multilevel_joint": "Multi-level, agents unfrozen", "multilevel_comm_joint": "Multi-level + comm., agents unfrozen"}
COLOR = {"central": "#7f7f7f", "multilevel": "#1f77b4", "multilevel_comm": "#e8710a",
         "multilevel_joint": "#1f77b4", "multilevel_comm_joint": "#e8710a"}
MARK = {"central": "s", "multilevel": "o", "multilevel_comm": "^"}
STYLE = {"central": "-", "multilevel": "-", "multilevel_comm": "--"}
HATCH = {"central": "..", "multilevel": "", "multilevel_comm": "//"}
MAIN = ["central", "multilevel", "multilevel_comm"]
LESIONS = [("modules1", "Single\nagent"), ("pair_conn", "Pair\nconnectors"), ("top_conn", "Top\nconnector")]
CORE_ENVS = ["in_dist", "wind", "sand", "slope_up"]


def load(tag):
    df = pd.read_csv(os.path.join("experiments", "results", tag, "results.csv"))
    ln = pd.read_csv("figures/csvs/body_lengths.csv").set_index("id").length
    df["body_len"] = df.group.map(lambda g: np.mean([ln[int(i)] for i in g.split("-")]))
    df["perf_bl"] = df.perf / df.body_len
    df["unit"] = df.seed.astype(str) + "_" + df.robot.astype(str)
    df["lesion_id"] = np.where(df.lesion == "modules", "modules" + df.k.astype(str), df.lesion)
    return df


def ratio_seed(d, lesion_id, envs):
    """Retention of one seed's rows `d` (one condition): mean over envs of ratio of means."""
    r = []
    for e in envs:
        de = d[d.env == e]
        base = de[de.lesion_id == "modules0"].perf.mean()
        r.append(de[de.lesion_id == lesion_id].perf.mean() / base)
    return float(np.mean(r))


def per_seed(df, cond, lesion_id, envs):
    return np.array([ratio_seed(g, lesion_id, envs) for _, g in df[df.cond == cond].groupby("seed")])


def ci(x):
    """mean, half-width of the 95% t-interval across seeds."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return x.mean(), np.nan
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def intact_seed(df, cond, envs, col="perf"):
    return np.array([g[(g.lesion_id == "modules0") & g.env.isin(envs)].groupby("env")[col].mean().mean()
                     for _, g in df[df.cond == cond].groupby("seed")])


def boot_ci(df, cond, lesion_id, envs, n=500, seed=0):
    """Secondary check: bootstrap over test robots (seed x grouping), pooling seeds."""
    rng = np.random.default_rng(seed)
    d = df[df.cond == cond]
    units = d.unit.unique()
    v = []
    for _ in range(n):
        pick = rng.choice(units, len(units))
        w = pd.Series(pick).value_counts()
        rs = []
        for e in envs:
            de = d[d.env == e]
            def wmean(sub):
                m = sub.groupby("unit").perf.mean()
                return np.average(m[w.index], weights=w.values)
            rs.append(wmean(de[de.lesion_id == lesion_id]) / wmean(de[de.lesion_id == "modules0"]))
        v.append(np.mean(rs))
    return np.percentile(v, [2.5, 97.5])


def choose_envs(df):
    """PLAN.md rule: add step_down only if mean intact distance for central AND multilevel is each >= 0.10."""
    envs = list(CORE_ENVS)
    note = "step_down not present in data"
    if "step_down" in df.env.unique():
        m = {c: df[(df.cond == c) & (df.env == "step_down") & (df.lesion_id == "modules0")].perf.mean()
             for c in ["central", "multilevel"]}
        ok = all(v >= 0.10 for v in m.values())
        note = f"step_down intact distance central {m['central']:.3f}, multilevel {m['multilevel']:.3f} -> " + \
               ("INCLUDED" if ok else "EXCLUDED (below the 0.10 rule)")
        if ok:
            envs.append("step_down")
    return envs, note


def style_axes(plt):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6, "xtick.major.size": 2,
                         "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def fmt_envs(envs):
    return [e.replace("_", "\n") for e in envs]


def main(tag, p2tag=None):
    df = load(tag)
    conds = [c for c in MAIN if c in df.cond.unique()]
    envs, env_note = choose_envs(df)
    nseeds = df.seed.nunique()
    out = os.path.join("experiments", "results", tag)
    style_axes(plt)

    # ---------------- main figure
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.0), gridspec_kw={"width_ratios": [1, 1]})
    ks = [0, 1, 2, 3]
    for c in conds:
        ps = [np.ones(nseeds)] + [per_seed(df, c, f"modules{k}", envs) for k in ks[1:]]
        m, h = zip(*[ci(x) for x in ps])
        m, h = np.array(m) * 100, np.array(h) * 100
        ax[0].plot(ks, m, STYLE[c], marker=MARK[c], color=COLOR[c], label=LABEL[c], ms=3.5, lw=1.2)
        ax[0].fill_between(ks, m - np.nan_to_num(h), m + np.nan_to_num(h), color=COLOR[c], alpha=0.2, lw=0)
    ax[0].set_xticks(ks, ["0\n(intact)", "1", "2", "3"])
    ax[0].set_xlabel("Modules disabled (of 4)")
    ax[0].set_ylabel("Locomotion retained\n(% of intact)")
    ax[0].set_ylim(0, 105)
    ax[0].set_title("A", loc="left", fontweight="bold", pad=2)
    w = 0.8 / len(conds)
    for i, c in enumerate(conds):
        ms_, hs_ = zip(*[ci(per_seed(df, c, l, envs)) for l, _ in LESIONS])
        ax[1].bar(np.arange(len(LESIONS)) + (i - (len(conds) - 1) / 2) * w, np.array(ms_) * 100, w, color=COLOR[c],
                  hatch=HATCH[c], edgecolor="white" if HATCH[c] else COLOR[c], linewidth=0,
                  yerr=np.nan_to_num(np.array(hs_)) * 100, capsize=1.5, error_kw=dict(lw=0.7))
    ax[1].set_xticks(range(len(LESIONS)), [n for _, n in LESIONS])
    ax[1].set_xlabel("Lesion")
    ax[1].set_ylim(0, 105)
    ax[1].set_title("B", loc="left", fontweight="bold", pad=2)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, handlelength=1.8, columnspacing=1.0,
               handletextpad=0.4, borderaxespad=0.1)
    fig.tight_layout(rect=(0, 0.11, 1, 1), pad=0.3, w_pad=0.6)
    fig.savefig(os.path.join(out, "levels_figure.png"), dpi=300)
    fig.savefig(os.path.join(out, "levels_figure.pdf"))
    plt.close(fig)

    # ---------------- supplementary: k=1 per environment
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    for i, c in enumerate(conds):
        vals = []
        for e in envs:
            r = [ratio_seed(g, "modules1", [e]) for _, g in df[df.cond == c].groupby("seed")]
            vals.append(ci(r))
        m, h = zip(*vals)
        ax.bar(np.arange(len(envs)) + (i - (len(conds) - 1) / 2) * w, np.array(m) * 100, w, color=COLOR[c],
               hatch=HATCH[c], edgecolor="white" if HATCH[c] else COLOR[c], linewidth=0,
               yerr=np.nan_to_num(np.array(h)) * 100, capsize=1.5, error_kw=dict(lw=0.7), label=LABEL[c])
    ax.set_xticks(range(len(envs)), fmt_envs(envs))
    ax.set_ylabel("Retained after one\nmodule lost (%)")
    ax.set_ylim(0, 105)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "levels_supp_env.png"), dpi=300)
    fig.savefig(os.path.join(out, "levels_supp_env.pdf"))
    plt.close(fig)

    # ---------------- summary numbers
    L = [f"# Summary: {tag}", "", f"Seeds: {nseeds}. Environments used: {envs}. {env_note}.",
         "Mean +/- 95% CI half-width across seeds (t-interval). Retention in % of intact.", ""]
    L += ["## Intact distance", "| condition | raw distance | agent body lengths |", "|---|---|---|"]
    for c in conds:
        a, b = ci(intact_seed(df, c, envs)), ci(intact_seed(df, c, envs, "perf_bl"))
        L.append(f"| {c} | {a[0]:.3f} +/- {a[1]:.3f} | {b[0]:.2f} +/- {b[1]:.2f} |")
    L += ["", "## Retention by number of modules disabled (%)", "| condition | k=1 | k=2 | k=3 |", "|---|---|---|---|"]
    for c in conds:
        cells = []
        for k in [1, 2, 3]:
            m, h = ci(per_seed(df, c, f"modules{k}", envs))
            cells.append(f"{100 * m:.1f} +/- {100 * h:.1f}")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "## Retention by lesion type (%)", "| condition | single agent | pair connectors | top connector |", "|---|---|---|---|"]
    for c in conds:
        cells = []
        for l, _ in LESIONS:
            m, h = ci(per_seed(df, c, l, envs))
            cells.append(f"{100 * m:.1f} +/- {100 * h:.1f}")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    if set(MAIN) <= set(conds):
        c1, m1, mc = (per_seed(df, c, "modules1", envs) for c in ["central", "multilevel", "multilevel_comm"])
        r = m1 / c1
        L += ["", "## H1 / H2 at k=1 (per-seed paired)",
              f"- multilevel / central retention ratio per seed: {np.round(r, 2).tolist()}  (mean {r.mean():.2f})",
              f"- multilevel - central (pct points): mean {100 * ci(m1 - c1)[0]:.1f}, 95% CI +/- {100 * ci(m1 - c1)[1]:.1f}",
              f"- multilevel_comm - multilevel (pct points): mean {100 * ci(mc - m1)[0]:.1f}, 95% CI +/- {100 * ci(mc - m1)[1]:.1f}"]
        L += ["", "## Secondary check: bootstrap over test robots, pooled across seeds (k=1, 95% interval, %)"]
        for c in conds:
            lo, hi = boot_ci(df, c, "modules1", envs)
            L.append(f"- {c}: [{100 * lo:.1f}, {100 * hi:.1f}]")
        L += ["", "## Per-seed k=1 retention (%)"]
        for c in conds:
            L.append(f"- {c}: {np.round(100 * per_seed(df, c, 'modules1', envs), 1).tolist()}")
    # P2 (only if available)
    if p2tag and os.path.exists(os.path.join("experiments", "results", p2tag, "results.csv")):
        d2 = load(p2tag)
        L += ["", f"## P2 ({p2tag}): agent MLPs unfrozen, 120 iterations"]
        L += ["| condition | intact distance | k=1 | single agent | pair conn | top conn |", "|---|---|---|---|---|---|"]
        res = {}
        for c in ["multilevel_joint", "multilevel_comm_joint"]:
            if c in d2.cond.unique():
                res[c] = per_seed(d2, c, "modules1", envs)
                i = ci(intact_seed(d2, c, envs))
                cells = [f"{100 * ci(per_seed(d2, c, l, envs))[0]:.1f} +/- {100 * ci(per_seed(d2, c, l, envs))[1]:.1f}" for l in ["modules1", "modules1", "pair_conn", "top_conn"][1:]]
                k1 = ci(res[c])
                L.append(f"| {c} | {i[0]:.3f} +/- {i[1]:.3f} | {100 * k1[0]:.1f} +/- {100 * k1[1]:.1f} | " + " | ".join(cells) + " |")
        if len(res) == 2:
            diff = res["multilevel_comm_joint"] - res["multilevel_joint"]
            m, h = ci(diff)
            L.append(f"\nPrimary P2 comparison: k=1 retention, comm_joint - multilevel_joint = {100 * m:.1f} pct points, "
                     f"95% CI +/- {100 * h:.1f} (benefit only if the CI excludes 0). Per-seed: {np.round(100 * diff, 1).tolist()}")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--tag", default="v2")
    p.add_argument("--p2tag", default="v2_p2")
    a = p.parse_args()
    main(a.tag, a.p2tag)
