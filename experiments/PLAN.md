# Pilot v2: pre-specification (written 2026-10-11, before any v2 run)

Purpose: one compact figure for the GRFP research plan. H1: multi-level control is more robust to module loss than a single central controller. H2 (open question): does communication between levels add robustness? Report whatever the runs produce; do not tune until communication "wins". Every variant tried is logged in `VARIANT_LOG.md`, including failures.

Baseline for comparison: `results/v1_baseline/` (3 seeds, 5 envs incl. low_gravity, old condition names).

## Conditions (names used in code, CSV and figure)
| name | figure label | what it is |
|---|---|---|
| `central` | Single central controller | one MLP over all 44 points, trained from scratch per robot (as v1 `flat`) |
| `multilevel` | Multi-level, no communication | 4 frozen agent MLPs + local connector MLP (fine-tuned) (as v1 `multi`) |
| `multilevel_comm` | Multi-level + communication | `multilevel` + two-scalar inter-level messages (as v1 `multi_comm`) |
| `multilevel_joint` (P2) | (supplementary) | `multilevel` with agent MLPs unfrozen, 120 iterations |
| `multilevel_comm_joint` (P2) | (supplementary) | `multilevel_comm` with agent MLPs unfrozen, 120 iterations |

Training for the three main conditions is unchanged from v1: 60 iterations, clip 0.08, 1000 steps, 12 groups per seed, flat ground. Seeds 0-2 reuse the v1 trained parameters (same budget and code path, only renamed); seeds 3-4 are newly trained. All seeds are re-evaluated with the v2 environments and lesions.

## Environments
`in_dist`, `wind`, `sand`, `slope_up` (low_gravity dropped: intact travel is dominated by the +0.15 initial velocity kick). Optional fifth: `step_down`. Decision rule fixed now: include `step_down` in the figure only if mean intact distance for `central` AND `multilevel` is each >= 0.10 (robots actually traverse it); otherwise report it as excluded in the variant log. Rule is applied once, to the results.

## Analysis (fixed now)
- Performance = mean x-displacement of all mass points after 1000 steps. Also reported in agent body lengths (distance divided by mean `figures/csvs/body_lengths.csv` length of the 4 agents in each robot).
- Retention for a seed = mean over environments of [mean perf with the lesion / mean perf intact] (ratio of means over that seed's 12 test robots and, for k modules, over all subsets of size k).
- Headline statistic: mean across seeds of retention, 95% CI across seeds (t-interval, n seeds - 1 d.o.f.). Bootstrap over test robots is a secondary check only, not shown in the figure.
- k = 4 (everything disabled) is not plotted.
- Module lesion = zero actuation of that agent's springs and of every connector touching it (control lesion, not physical removal).

## Pre-specified comparisons and expectations
1. **H1, parity**: intact distance should be similar across the three main conditions (v1: 0.229 / 0.218 / 0.225).
2. **H1, robustness**: retention at k=1, `multilevel` vs `central`. Expectation: multilevel higher (v1: about 0.50 vs 0.27). I expect the sign to replicate; the size may move with 5 seeds. H1 is supported only if the across-seed CIs do not overlap or the paired per-seed difference CI excludes 0.
3. **H2 (main conditions)**: retention at k=1, `multilevel_comm` vs `multilevel`. Expectation: no measurable difference (v1: 0.49 vs 0.50). Absence of a difference is the expected outcome; it will be reported as such.
4. **P1, level-specific lesions** (evaluation only, all conditions, all seeds), disabling only: (a) the level-3 connector (1,2); (b) the two within-pair connectors (0,1) and (2,3); (c) one agent (the k=1 module lesion). Expectation: for the multi-level conditions the level-3 connector lesion costs least and the within-pair connector lesion costs more, with single-agent loss the largest, because connector springs carry much of the thrust and there are two pair connectors versus one top connector. For `central` I expect a similar ordering but no principled reason for a difference between levels (it has no level structure). If the level-3 lesion costs about nothing in every condition, the interpretation is that the third level is barely exercised in this setup.
5. **P2, one stronger-communication attempt**, run once as written here: unfreeze the agent MLPs (per-robot weights, trained on the test robots like `central`, since unfrozen agent weights are per body), 120 iterations, same clip, 12 groups per seed, 5 seeds; apply the identical regime (`multilevel_joint`) to the no-communication comparator. Scalar two-message signal as in v1 (richer vector signal not attempted). Primary comparison: per-seed difference in k=1 retention, `multilevel_comm_joint` minus `multilevel_joint`; it counts as a benefit only if the 95% CI across seeds excludes 0. Expectation: probably no benefit; I have no strong prior. Report either way; no second attempt.

## Out of scope (future work only)
Physical module removal, more than three levels, central controller initialized from pretrained agent weights, richer learned message vectors.

## Caveats to state
Control lesion rather than physical removal; short training budget; small number of seeds; central controller trained on the exact test robots (favors central).
