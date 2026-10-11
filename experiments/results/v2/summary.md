# Summary: v2

Seeds: 5. Environments used: ['in_dist', 'wind', 'sand', 'slope_up', 'step_down']. step_down intact distance central 0.297, multilevel 0.245 -> INCLUDED.
Mean +/- 95% CI half-width across seeds (t-interval). Retention in % of intact.

## Intact distance
| condition | raw distance | agent body lengths |
|---|---|---|
| central | 0.274 +/- 0.030 | 1.71 +/- 0.18 |
| multilevel | 0.249 +/- 0.037 | 1.55 +/- 0.23 |
| multilevel_comm | 0.245 +/- 0.043 | 1.53 +/- 0.26 |

## Retention by number of modules disabled (%)
| condition | k=1 | k=2 | k=3 |
|---|---|---|---|
| central | 23.1 +/- 3.0 | 7.0 +/- 1.1 | 2.4 +/- 0.4 |
| multilevel | 40.1 +/- 3.1 | 14.3 +/- 1.2 | 4.0 +/- 0.9 |
| multilevel_comm | 40.6 +/- 3.7 | 14.4 +/- 1.0 | 4.0 +/- 1.0 |

## Retention by lesion type (%)
| condition | single agent | pair connectors | top connector |
|---|---|---|---|
| central | 23.1 +/- 3.0 | 9.6 +/- 3.5 | 26.2 +/- 5.4 |
| multilevel | 40.1 +/- 3.1 | 29.1 +/- 5.4 | 79.7 +/- 13.6 |
| multilevel_comm | 40.6 +/- 3.7 | 30.3 +/- 7.5 | 80.6 +/- 14.2 |

## H1 / H2 at k=1 (per-seed paired)
- multilevel / central retention ratio per seed: [1.85, 1.79, 1.68, 1.43, 2.02]  (mean 1.76)
- multilevel - central (pct points): mean 17.1, 95% CI +/- 4.8
- multilevel_comm - multilevel (pct points): mean 0.5, 95% CI +/- 4.9

## Secondary check: bootstrap over test robots, pooled across seeds (k=1, 95% interval, %)
- central: [20.5, 25.3]
- multilevel: [37.3, 42.9]
- multilevel_comm: [37.4, 43.2]

## Per-seed k=1 retention (%)
- central: [21.5, 21.0, 24.3, 26.8, 21.8]
- multilevel: [39.9, 37.6, 40.8, 38.4, 44.0]
- multilevel_comm: [39.0, 37.8, 40.0, 45.6, 40.7]

## P2 (v2_p2): agent MLPs unfrozen, 120 iterations
| condition | intact distance | k=1 | single agent | pair conn | top conn |
|---|---|---|---|---|---|
| multilevel_joint | 0.412 +/- 0.035 | 36.2 +/- 3.1 | 36.2 +/- 3.1 | 27.1 +/- 5.8 | 70.6 +/- 17.6 |
| multilevel_comm_joint | 0.472 +/- 0.046 | 34.6 +/- 3.7 | 34.6 +/- 3.7 | 24.3 +/- 5.5 | 54.6 +/- 13.9 |

Primary P2 comparison: k=1 retention, comm_joint - multilevel_joint = -1.6 pct points, 95% CI +/- 5.0 (benefit only if the CI excludes 0). Per-seed: [-2.4, 5.2, -2.2, -2.9, -5.7]
