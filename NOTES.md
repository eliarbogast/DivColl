# DivColl code notes (for future sessions)

Written from a read-through of the code, not from the paper. Verify against the code before relying on it.

## What it is
Differentiable 2D mass-spring robots (Taichi, runs on Metal here: `ti.init(arch=ti.gpu)`).
Taichi simulation code in `simulator/sim.py` (ELDiR-derived, patched). Python 3.10, use `.venv`.

## Agent (individual robot)
- 11 mass points, 21 springs (`operators/defaults/config.py`). Springs have rest length, stiffness, actuation amplitude.
- Controller: one MLP per agent. Input = 10 sin waves (omega=10) + (rel. pos x/y, vel x/y) per mass point (4*11). 32 hidden, tanh. Output = one activation per spring (smoothed: 0.85 old + 0.15 new). Spring target length = rest * (1 + actuation * act).
- Weights shapes in `outputs/og_bots/N.pkl`: w1 (1,32,54), w2 (1,21,32).
- Loss = -(mean x displacement of all mass points) after `sim_steps` (default 1000, dt=0.004). Evaluation score = -loss.
- Agents are trained by evolution (body + spring genome, `train_components.py`) with gradient descent on NN weights inside each evaluation (15 iterations, clipped, `update_weights`).

## Collective
- `create_collective.py` merges N agents into one "robot" in one sim row: points/springs concatenated, `p_id`/`s_id` mark ownership (1..N). Agents are shifted along x by `placing[...]` offsets (0.3, 0, -0.3, -0.6, -0.9: a tree-shaped placement defined for up to 25 agents). Each agent keeps its OWN NN (weights1[r, s, ...] indexed by sub-bot s).
- Connectors: 5 springs between a pair of agents (anchors = feet/top points found from a 1000-step flat-ground rollout: `identify_connector_anchors`), stiffness 2000, actuation 0.6, `s_id` negative = connector. Parent->children links come from `placing[b][1]` (tree structure).
- Connector controller (`conn_*` fields): ONE shared MLP, 12 inputs per connector spring (rel. pos and vel of its two anchors, touch at both anchors, [act, always 0, known bug noted in code], rest length), 32 hidden, 1 output per connector spring. It only sees its own anchors, so it is purely local.
- Agent NNs never see anything about each other, and receive no signal from connectors. The ONLY inter-agent coupling is physical (spring forces).
- Only connectors are trained (`train_connector.py` -> `simulate_conn` -> `optimize_conn`); agent weights are frozen. Trained on pairs only. `fill_weights_c` copies the pair-trained connector weights into every connector slot of larger collectives (that is how sizes 5/10/20 in `larger_colls` work).
- Evaluate: `evaluate.py` (collect_bots -> `forward_visualization` -> `loss_per_bot.npy`, `loss.npy`). Per-agent score = mean x displacement of that agent's points.

## Gotchas
- `evaluate.py` and `train_connector.py` call `collect_bots`, which WRITES `<bot_dir>/bots_<k>.pkl`. For `outputs/og_bots` that is a tracked file, so running evaluate there dirties the git tree (`git checkout outputs/og_bots/bots_0.pkl`). Use a scratch dir for new experiments.
- `evaluate.py` and `forward_visualization` write state to `tmp/state/*.npy`.
- Sim globals are module-level; one `ti.init` per process, so run each condition/seed batch as a separate process or reuse one init.
- `sim_steps` is a global set by `env_setup`. GPU arrays are allocated per `setup()` call.
- `evaluate.py` hard-codes ffmpeg path /opt/homebrew/bin/ffmpeg (only matters for video).
- Speed: evaluating 4 agents for 1000 steps, including Taichi startup, took ~15 s wall.
- Environments (`evaluate.py: setup_sim`, `sim.py: env_setup`): in_dist, slopes, steps, gap, obs, ice, sticky, wind, sand, low_gravity, treadmill.
- Figures: `figures/*.py` read `figures/csvs/` only. Style reference for new figures.
