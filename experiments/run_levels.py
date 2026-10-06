"""Flat vs multi-level vs multi-level+communication: training, ablation evaluation, results CSV.

Workers (one Taichi init per process):
    python experiments/run_levels.py train --cond multi_comm --seed 0
    python experiments/run_levels.py eval  --cond multi_comm --seed 0 --env wind
Orchestrator (spawns the workers):
    python experiments/run_levels.py all --seeds 0 1 2
Results: experiments/results/<tag>/results.csv  (one row per robot x ablation); plot with plot_levels.py.
"""
import os, sys, pickle, subprocess, time
from argparse import ArgumentParser
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from experiments.levels_lib import *

TRAIN_POOL, TEST_POOL = np.arange(0, 50), np.arange(50, 100)


def groups_for(cond, seed, n_groups, role):
    """role='test' groups use held-out agents (50-99); multi* train on agents 0-49. flat trains on the test robots
    themselves (its controller is per-robot, as the agent MLPs were)."""
    if role == "test" or cond == "flat":
        return make_groups(TEST_POOL, n_groups, np.random.default_rng(seed))
    return make_groups(TRAIN_POOL, n_groups, np.random.default_rng(1000 + seed))


def conn_file_for(cond, seed):
    return None if cond == "flat" else os.path.join(ROOT, "outputs", "og_connectors", "div", str(seed % 10), "best.pkl")


def paths(tag, cond, seed):
    d = os.path.join(ROOT, "experiments", "results", tag)
    os.makedirs(d, exist_ok=True)
    return d, os.path.join(d, f"params_{cond}_{seed}.pkl"), os.path.join(ROOT, "experiments", "scratch", tag)


def worker_train(a):
    d, pfile, scratch = paths(a.tag, a.cond, a.seed)
    groups = groups_for(a.cond, a.seed, a.n_groups, "train")
    bundle = os.path.join(scratch, f"train_{a.cond}_{a.seed}.pkl")
    write_bundle(groups, "flat" if a.cond == "flat" else "multi", bundle)
    S = init_sim(bundle, "in_dist", a.cond, a.steps, conn_file_for(a.cond, a.seed), a.seed, a.clip)
    t0 = time.time()
    snap, hist = train(S, a.cond, a.iters)
    print(f"trained {a.cond} seed {a.seed} in {time.time() - t0:.0f}s", flush=True)
    with open(pfile, "wb") as f:
        pickle.dump(dict(snap=snap, hist=hist, groups=groups), f)


def worker_eval(a):
    d, pfile, scratch = paths(a.tag, a.cond, a.seed)
    with open(pfile, "rb") as f:
        saved = pickle.load(f)
    groups = groups_for(a.cond, a.seed, a.n_groups, "test")
    bundle = os.path.join(scratch, f"test_{a.cond}_{a.seed}.pkl")
    b = write_bundle(groups, "flat" if a.cond == "flat" else "multi", bundle)
    S = init_sim(bundle, a.env, a.cond, a.steps, conn_file_for(a.cond, a.seed), a.seed)
    restore(S, saved["snap"])
    rows = []
    for sil, perf in evaluate(S, b["owner"]):
        for r, p in enumerate(perf):
            rows.append(dict(cond=a.cond, seed=a.seed, env=a.env, k=len(sil), silenced=",".join(map(str, sil)),
                             robot=r, group="-".join(map(str, groups[r])), perf=float(p)))
    out = os.path.join(d, "results.csv")
    pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)


def run_worker(args):
    cmd = [sys.executable, os.path.abspath(__file__)] + [str(x) for x in args]
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"worker failed: {' '.join(cmd)}")


def orchestrate(a):
    common = ["--tag", a.tag, "--steps", a.steps, "--n_groups", a.n_groups, "--clip", a.clip]
    for seed in a.seeds:
        for cond in a.conds:
            run_worker(["train", "--cond", cond, "--seed", seed, "--iters", a.iters] + common)
            for env in a.envs:
                run_worker(["eval", "--cond", cond, "--seed", seed, "--env", env] + common)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("mode", choices=["train", "eval", "all"])
    p.add_argument("--tag", default="main")
    p.add_argument("--cond", default="multi", choices=CONDITIONS)
    p.add_argument("--conds", nargs="+", default=CONDITIONS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--env", default="in_dist")
    p.add_argument("--envs", nargs="+", default=ENVS)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--n_groups", type=int, default=20)
    p.add_argument("--clip", type=float, default=0.04, help="gradient step size (same for all conditions)")
    a = p.parse_args()
    {"train": worker_train, "eval": worker_eval, "all": orchestrate}[a.mode](a)
