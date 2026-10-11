# Figure 1 text and numbers for the research plan (pilot v2)
Source: `summary.md` in this folder (5 training seeds, 5 environments: in_dist, wind, sand, slope_up, step_down; 12 held-out test robots per seed). Figure: `levels_figure.pdf` / `.png` (6.5 x 2.0 in).

## Caption (filled template)
**Figure 1.** Multi-level control preserved locomotion after module loss better than a single central controller, at similar intact performance (intact distance: 1.7 vs. 1.6 vs. 1.5 agent body lengths for central, multi-level, and multi-level + communication). After one module was disabled, multi-level robots retained 40% of intact locomotion vs. 23% for the central controller (1.8x). A minimal form of communication between levels did not measurably change performance. **(A)** Locomotion retained vs. number of modules disabled. **(B)** Retention after disabling one agent, both within-pair connectors, or only the top-level connector. Lines/bars show means across 5 training seeds; bands/error bars show 95% confidence intervals across seeds. "Disabled" = actuation removed (control lesion).

## Alt text
Two-panel figure. Left: line plot of locomotion retained, as a percentage of intact performance, versus the number of modules disabled (0 to 3) for three controllers. The single central controller declines fastest (about 23% retained after one module is disabled); the two multi-level controllers decline more slowly (about 40%) and overlap each other. Right: bar chart of locomotion retained for three lesion types (single agent, pair connectors, top-level connector). Multi-level controllers retain more than the central controller for every lesion type, most strikingly when only the top-level connector is disabled (about 80% versus 26%); the two multi-level controllers are indistinguishable.

## Numbers (mean +/- 95% CI across 5 seeds)
- Intact distance, agent body lengths: central 1.71 +/- 0.18, multilevel 1.55 +/- 0.23, multilevel_comm 1.53 +/- 0.26 (raw: 0.274 / 0.249 / 0.245). Central is ~10% higher (paired diff 0.026 +/- 0.027, CI includes 0), which if anything favours central.
- Retention after one module lost: central 23.1 +/- 3.0 %, multilevel 40.1 +/- 3.1 %, multilevel_comm 40.6 +/- 3.7 %.
- Ratio multilevel/central at k=1: mean 1.76 (per seed 1.85, 1.79, 1.68, 1.43, 2.02). Difference 17.1 +/- 4.8 percentage points. Replaces "about twice" with "about 1.8x".
- k=2: 7.0 / 14.3 / 14.4 %; k=3: 2.4 / 4.0 / 4.0 %.
- Communication (main, frozen agents): multilevel_comm - multilevel = +0.5 +/- 4.9 percentage points at k=1 (no effect).
- P1, lesion type (central / multilevel / multilevel_comm): single agent 23 / 40 / 41 %; pair connectors 10 / 29 / 30 %; top connector 26 / 80 / 81 %. Disabling the within-pair connectors hurt most for every controller; disabling the top-level connector hurt multi-level robots least (they keep ~80%) but cost the central controller about as much as losing an agent.
- P2 (agent MLPs trained jointly with communication, 120 iterations, single pre-specified attempt): k=1 retention 36.2 +/- 3.1 % (no communication) vs 34.6 +/- 3.7 % (communication); difference -1.6 +/- 5.0 points -> no benefit. Exploratory: intact distance was higher with communication (+0.060 +/- 0.032, ~15%), not accompanied by better retention.
- Secondary check (bootstrap over test robots, k=1): central [20.5, 25.3] %, multilevel [37.3, 42.9] %, multilevel_comm [37.4, 43.2] %.

## Caveats (one clause each)
Control lesion rather than physical removal; short training budget and 5 seeds; central controller trained on the exact test robots (favours central); communication tested in one minimal form (two scalars per level).
