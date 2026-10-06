# Three-level robot experiment: findings and handoff

Self-contained summary for another agent. Repo: `/Users/eli/Documents/GitHub/DivColl` (user's fork of the DivColl code, ELDiR-derived Taichi mass-spring simulator). Read `NOTES.md` in the repo root for how the simulator works. Nothing here has been committed to git.

## Purpose
Preliminary evidence for a project proposal (deadline around October 13, 2026). The aim is one informative figure comparing three conditions on a 4-agent robot with three levels (agents, pairs, whole robot):
1. **flat**: one centralized controller over the whole robot
2. **multi**: independent agent controllers joined by local connectors (essentially Hein-style tethering), no communication between levels
3. **multi_comm**: the same hierarchy plus inter-level communication

The metric is how much performance each condition retains when modules are disabled. The user wants compact figures and has agreed to the defaults below.

## What was built
- `experiments/levels_lib.py`: builds the 4-agent robots and the flat variant, plus training loops and ablation evaluation. Reuses `create_collective.py` anchor logic.
- `experiments/run_levels.py`: `train`, `eval` and `all` modes. Each worker process initializes Taichi once (per-env constants are baked at kernel compile time), so every train or eval run is its own process. Run from the repo root.
- `experiments/plot_levels.py`: figure from `results.csv` only, no simulation. 6.8 x 2.1 in, PNG at 300 dpi plus PDF.
- `experiments/data/conn_points.csv`: connector anchor points for the 100 pretrained agents (generated once).
- `simulator/sim.py`, modified behind a flag that is off by default. Additions:
  - `comm_enabled`
  - the `comm_signals` kernel
  - `update_weights_comm`
  - `init_comm`
  - new comm fields
  - edits to `nn1`, `clear_states`, `clear_grad`, `forward` and `manual_backward`

  `patches/eldir-to-divcoll.patch` no longer reproduces this file exactly.
- `.gitignore` gained `experiments/scratch/` and `experiments/results/*/params_*.pkl`.
- `NOTES.md` (repo root): code notes.
- `experiments/results/smoke/`: leftover smoke-test output, safe to delete.
- Results live in `experiments/results/main/`: `results.csv` (8,640 rows), `levels_figure.png/.pdf`, `params_*.pkl` (gitignored). The run log is `experiments/run_main.log`.

## Design decisions (all defaults the user approved)
- **Levels**: level 1 is the individual agent (11 points, 21 springs, own frozen MLP, taken from `outputs/og_bots`). Level 2 is two pairs, (0,1) and (2,3). Level 3 is the whole robot, with a connector (1,2) between the pairs. Connectors are the paper's 5-spring connectors (stiffness 2000, actuation 0.6).
- **flat**: identical bodies and springs, with the connector springs actuated by the same single MLP. Controller trained from scratch, per robot, on the test robots themselves. This favors flat.
- **multi**: pretrained agent MLPs frozen; the shared connector MLP is initialized from `outputs/og_connectors/div/<seed%10>/best.pkl` and fine-tuned.
- **multi_comm**: multi, plus communication.
  - Each pair pools mean x-velocity and touch fraction over its points.
  - A level-3 controller reads both pools and emits `m3`.
  - Each level-2 controller reads its pool plus `m3` and emits `m2`.
  - Every agent's hidden layer receives `(m2 of its pair, m3)` through new weights initialized to zero.
  - The controller weights are small random.
  - Parameters are shared across robots.
- **Training**: gradient descent through the differentiable physics, normalized-gradient step (clip 0.08) for all conditions, 60 iterations, 1000 steps, flat ground. The best iterate is kept (per robot for flat; best mean loss for the shared multi parameters). Train on agents 0-49 and test on held-out agents 50-99 for multi and multi_comm. 12 groups of 4 distinct agents per seed, 3 seeds.
- **Evaluation environments**: in_dist, wind, sand, slope_up, low_gravity. The score is mean x-displacement of all mass points (higher is better).
- **Ablation**: disabling module k zeroes the actuation amplitude of agent k's springs and of every connector touching agent k. The body stays as a passive mass. All subsets of size 0-4 were evaluated (size 4 is the passive floor, near 0).
- **Retained** = mean perf with k modules disabled divided by mean intact perf. This is a ratio of means within each environment, averaged over subsets and then over environments. CIs bootstrap over test robots (seed x grouping). They therefore understate uncertainty from training runs, since there are only 3 seeds.

## Results (3 seeds x 5 envs, means over environments)
| | flat | multi | multi_comm |
|---|---|---|---|
| Intact distance | 0.229 | 0.218 | 0.225 |
| Retained, 1 module disabled | 0.27 | 0.50 | 0.49 |
| Retained, 2 disabled | 0.10 | 0.22 | 0.22 |
| Retained, 3 disabled | 0.05 | 0.08 | 0.08 |

- **Intact performance is matched across conditions.** This neutralizes the "flat is just a worse walker" objection. In training, flat from scratch reached about 0.29-0.34 mean best versus 0.31-0.33 for multi.
- **Flat vs multi-level is consistent across seeds.** Retention at k=1 was flat 0.27/0.27/0.28, multi 0.48/0.50/0.52 and multi_comm 0.48/0.51/0.49 across seeds 0/1/2. It holds in every environment except the degenerate low-gravity one. With the all-disabled floor subtracted, flat is 0.21 and multi is 0.39 (excluding low gravity).
- **Per-environment retention at k=1**:

  | | in_dist | wind | sand | slope_up | low_gravity |
  |---|---|---|---|---|---|
  | flat | 0.17 | 0.39 | 0.19 | 0.17 | 0.44 |
  | multi | 0.33 | 0.54 | 0.39 | 0.33 | 0.89 |
  | multi_comm | 0.33 | 0.57 | 0.38 | 0.31 | 0.87 |

- **Communication gave no measurable benefit.** The paired differences in raw performance were +0.007 +/- 0.005 intact and +0.002 +/- 0.001 with one module disabled.
- **The communication weights were only partly used.** `comm_in` moved off zero in 2 of 3 seeds. RMS was about 0.03 and 0.05 in seeds 0 and 2, versus about 0.05 for the connector weights. In seed 1 the best iterate was iteration 2, so communication barely engaged. Best iterates were iteration 52, 2 and 60 for multi_comm, and 58, 2 and 22 for multi. This is a null under a weak optimization budget, not evidence that communication cannot help.
- **Low gravity is a degenerate environment.** Intact travel is only about 0.04-0.06, driven mostly by the simulator's initial velocity kick (`+0.15` on step 1 in `advance`), so actuation matters little and retention is inflated (0.89 for multi). Recommend dropping it or flagging it.
- **Training is chaotic.** Gradients through contact physics are noisy; loss swings between iterations, so best-iterate selection matters.

## The figure (`levels_figure.png`)
Grey is flat, blue is multi, orange is multi_comm, with one shared legend below.
- **A, Intact:** grouped bars of distance travelled per environment, with 95% CIs across test robots. The conditions are indistinguishable apart from small differences in wind and in_dist.
- **B, Degradation:** retained performance against modules disabled (0 to 4). Flat falls fastest, the two multi-level curves overlap, and all reach about 0 by 3-4. The CI bands are very narrow because they bootstrap over test robots from only 3 seeds.
- **C, One module lost:** retained performance at k=1 per environment. Multi beats flat in all five; low gravity is the degenerate case.
- A suggested caption was drafted in the conversation. Its first sentence is: "Multi-level control preserves locomotion after module loss better than a flat controller, at equal intact performance; inter-level communication added no measurable benefit."

## Caveats the proposal should state
1. Small design: 3 seeds, a short 60-iteration budget, 12 test robots per seed.
2. Flat trains on the exact robots it is tested on, whereas multi trains a shared connector on other agents. This favors flat, so the intact parity is conservative.
3. "Disabled" means a control lesion (no actuation), not physical removal.
4. Only one communication design was tried (two scalars per level, frozen agent MLPs). A null here is weak evidence.
5. The ablation acts mostly at the agent level. The third level is only partly exercised.
6. The CIs reflect variation over test robots, not over independent training runs.

## Suggested next steps (priority order)
1. Train the agent MLPs jointly with the communication instead of freezing them.
2. Use richer inter-level signals, such as a learned vector instead of two scalars.
3. Add a level-specific lesion, for example cutting only the level-3 connector, to test whether the third level adds anything.
4. Use physical module removal and more seeds, and report variation across training runs.
5. Add environments that actually perturb the robot, replacing low gravity.
6. Add a flat variant initialized from the pretrained agent weights. The current simulator has one global hidden width of 32, so a block-structured initialization is not directly possible.

## Gotchas for whoever continues
- Run everything from the repo root, inside `.venv` (Python 3.10, Taichi 1.7.2 on Metal).
- Do not reuse a Taichi process across environments, because environment constants are baked at compile time.
- `evaluate.py` and `collect_bots` overwrite `outputs/og_bots/bots_0.pkl`. The user has confirmed overwriting existing outputs is fine, since the repo is their fork.
- Atomics make runs slightly non-deterministic. For example, iteration-0 performance differed by about 0.002 between multi and multi_comm.
- Reproduce the full run with `python experiments/run_levels.py all --tag main --seeds 0 1 2 --iters 60 --n_groups 12 --clip 0.08`. It takes about an hour. Regenerate the figure with `python experiments/plot_levels.py --tag main`.
