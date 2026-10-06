"""Shared code for the three-level robot experiment (flat vs multi-level vs multi-level + communication).

Every worker process must be started from the repo root (sim.py uses repo-relative imports) and calls
ti.init exactly once, so training and each evaluation environment run in separate processes.

Robot layout (4 agents, x offsets 0.3 / 0 / -0.3 / -0.6, agent 0 in front):
    level 1: agents 0,1,2,3 (own frozen MLP each)
    level 2: modules {0,1} and {2,3}  (joined by connectors 0-1 and 2-3)
    level 3: the whole robot          (modules joined by connector 1-2)
"""
import os, sys, pickle, itertools
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

N_AGENTS = 4
OFFSETS = [0.3, 0.0, -0.3, -0.6]          # same x offsets as create_collective.placing
EDGES = [(0, 1), (1, 2), (2, 3)]          # (0,1),(2,3) are level-2 links, (1,2) is the level-3 link
CONN_STIFFNESS, CONN_ACT = 2000, 0.6
BOT_DIR = os.path.join(ROOT, "outputs", "og_bots")
ANCHOR_CSV = os.path.join(ROOT, "experiments", "data", "conn_points.csv")
CONDITIONS = ["flat", "multi", "multi_comm"]
ENVS = ["in_dist", "wind", "sand", "slope_up", "low_gravity"]
TERRAIN_ENVS = ["slope_down", "obs", "slope_up", "gap", "step_down", "step_up"]


# ----------------------------------------------------------------------------- body construction
def load_bot(i):
    with open(os.path.join(BOT_DIR, f"{i}.pkl"), "rb") as f:
        return pickle.load(f)


def make_anchor_csv(n_bots=100):
    """Foot/top anchor points for each pretrained agent (same method as create_collective.py)."""
    from utils.eval_utils import collect_bots
    from simulator.sim import forward_visualization
    from create_collective import partition_bot, find_feet
    collect_bots(0, BOT_DIR, "", 1, 1, n_bots)
    forward_visualization(f"{BOT_DIR}/bots_0.pkl", "tmp", None, None, 1000)
    xs = np.load(os.path.join("tmp/state", "x.npy"))[:, :-1]
    rows = []
    for b in range(xs.shape[0]):
        left, right, top, _, std = partition_bot(xs, b)
        lf, rf = find_feet(b, left, right)
        nt = top * (1 - std)
        rows.append(dict(ID=b, left_foot=lf, left_top=(nt * left).argmax(),
                         right_foot=rf, right_top=(nt * right).argmax()))
    os.makedirs(os.path.dirname(ANCHOR_CSV), exist_ok=True)
    pd.DataFrame(rows).to_csv(ANCHOR_CSV, index=False)


def build_robot(ids, kind, anchors):
    """kind='multi': 4 agents with own MLPs + connector springs (negative s_id).
    kind='flat': identical bodies and springs, but all springs/points owned by ONE sub-bot (single MLP).
    Returns the robot dict and `owner` (agent index per spring; connector springs of edge e get -1-e)."""
    pts, spr, s_id, p_id, weights, owner = [], [], [], [], [], []
    bots = [load_bot(i) for i in ids]
    ppb = len(bots[0]["points"][0])
    for a, bot in enumerate(bots):
        start = len(pts)
        for p in bot["points"][0]:
            pts.append([p[0] + 0.05 + OFFSETS[a], p[1]])
            p_id.append(a + 1)
        for s in bot["springs"][0]:
            spr.append((s[0] + start, s[1] + start, s[2], s[3], s[4]))
            s_id.append(a + 1)
            owner.append(a)
        weights.append(bot["weights"])
    for e, (front, hind) in enumerate(EDGES):
        f = anchors[anchors["ID"] == ids[front]].iloc[0]
        h = anchors[anchors["ID"] == ids[hind]].iloc[0]
        pa = [f.left_foot] * 3 + [f.right_top, f.left_top]
        pb = [h.right_top, h.left_top, h.right_foot, h.right_foot, h.right_foot]
        for i, j in zip(pa, pb):
            i, j = int(i) + ppb * front, int(j) + ppb * hind
            length = float(np.linalg.norm(np.array(pts[i]) - np.array(pts[j])))
            spr.append([i, j, length, CONN_STIFFNESS, CONN_ACT])
            s_id.append(-(front + 1))
            owner.append(-1 - e)
    robot = dict(points=pts, springs=spr, weights=weights, ids=list(ids))
    if kind == "multi":
        robot.update(s_id=s_id, p_id=p_id)
    else:
        del robot["weights"]
    return robot, np.array(owner)


def make_groups(pool, n_groups, rng):
    """n_groups groups of N_AGENTS distinct agents drawn from `pool` (random order within a group)."""
    groups = []
    while len(groups) < n_groups:
        perm = rng.permutation(pool)
        for k in range(len(perm) // N_AGENTS):
            if len(groups) < n_groups:
                groups.append([int(i) for i in perm[k * N_AGENTS:(k + 1) * N_AGENTS]])
    return groups


def write_bundle(groups, kind, path):
    anchors = pd.read_csv(ANCHOR_CSV)
    robots, owners = zip(*[build_robot(g, kind, anchors) for g in groups])
    bundle = {k: [r[k] for r in robots] for k in robots[0]}
    # build_robot stores per-robot lists; the simulator expects the list-of-lists layout of collect_bots
    bundle["points"] = [r["points"] for r in robots]
    bundle["springs"] = [r["springs"] for r in robots]
    bundle["owner"] = list(owners)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


# ----------------------------------------------------------------------------- simulator driver
def env_files(env):
    ground = os.path.join(ROOT, "terrains", f"{env}.npy") if env in TERRAIN_ENVS else None
    cond = env if env in ["ice", "sticky", "wind", "sand", "low_gravity", "treadmill"] else None
    return ground, cond


def init_sim(bundle_path, env, condition, steps, conn_file=None, seed=0, clip=None):
    """ti.init + load robots. Returns the sim module. Call once per process."""
    import taichi as ti
    import simulator.sim as S
    ti.init(default_fp=ti.f32, arch=ti.gpu, random_seed=seed, device_memory_fraction=0.99)
    S.set_ti_globals()
    if clip is not None:
        S.gradient_clip = clip  # normalized-gradient step size, baked in when kernels first compile
    ground, cond = env_files(env)
    S.comm_enabled = condition == "multi_comm"
    S.env_setup(cond, steps, 1 if condition == "flat" else N_AGENTS)
    S.setup(bundle_path, ground, conn_file)
    if S.comm_enabled:
        S.init_comm(seed)
    return S


def trainable(S, condition):
    if condition == "flat":
        return ["weights1", "bias1", "weights2", "bias2"]
    names = ["conn_weights1", "conn_bias1", "conn_weights2", "conn_bias2"]
    if condition == "multi_comm":
        names += ["comm_in", "comm_w2", "comm_b2", "comm_w3", "comm_b3"]
    return names


def snapshot(S, names):
    return {n: getattr(S, n).to_numpy() for n in names}


def restore(S, snap):
    for n, a in snap.items():
        getattr(S, n).from_numpy(a)


def train(S, condition, iters, log=print):
    """Gradient training of the condition's trainable parameters (agent MLPs frozen for multi/multi_comm).
    flat: per-robot weights, best iterate kept per robot.  multi*: shared parameters, best mean loss kept.
    Returns (snapshot of best parameters, loss history array [iters+1, n_robots])."""
    names = trainable(S, condition)
    R = S.n_robots
    hist = np.full((iters + 1, R), np.nan)
    best = np.full(R, np.inf) if condition == "flat" else np.inf
    snap = snapshot(S, names)
    for k in range(iters + 1):
        S.clear()
        S.forward()
        L = S.loss.to_numpy()
        hist[k] = L
        bad = ~np.isfinite(L)
        if condition == "flat":
            better = (L < best) & ~bad
            if better.any():
                cur = snapshot(S, names)
                for n in names:
                    snap[n][better] = cur[n][better]
                best[better] = L[better]
        else:
            if bad.any():
                log(f"  non-finite loss at iter {k}; stopping and keeping best")
                break
            if L.mean() < best:
                best, snap = L.mean(), snapshot(S, names)
        if k % 10 == 0 or k == iters:
            log(f"  iter {k:3d} perf mean {-L[~bad].mean():.4f}  min {-L[~bad].max():.4f}  max {-L[~bad].min():.4f}")
        if k == iters:
            break
        S.loss.grad.fill(1.0)
        S.manual_backward()
        if condition == "flat":
            S.update_weights()
        else:
            S.update_weights_conn()
            if condition == "multi_comm":
                S.update_weights_comm()
    return snap, hist


def ablation_sets(n=N_AGENTS, max_k=N_AGENTS):
    return [c for k in range(max_k + 1) for c in itertools.combinations(range(n), k)]


def evaluate(S, owner, log=print):
    """Per-robot performance (mean x displacement of all points) for every disabled-module subset.
    Disabling module k = zero actuation amplitude on agent k's springs AND on every connector touching agent k
    (the bodies stay as passive masses). k=4 disables everything: the passive floor."""
    base = S.spring_actuation.to_numpy()
    rows = []
    owner_pad = np.full(base.shape, -9)
    for r, o in enumerate(owner):
        owner_pad[r, :len(o)] = o
    for sil in ablation_sets():
        a = base.copy()
        dead_owner = list(sil) + [-1 - e for e, ends in enumerate(EDGES) if set(ends) & set(sil)]
        a[np.isin(owner_pad, dead_owner)] = 0.0
        S.spring_actuation.from_numpy(a)
        S.clear()
        S.forward()
        perf = -S.loss.to_numpy()
        rows.append((sil, perf))
        log(f"  disabled {sil!s:14} mean perf {np.nanmean(perf):.4f}")
    S.spring_actuation.from_numpy(base)
    return rows
