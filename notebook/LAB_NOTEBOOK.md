# Lab notebook

Written by the loop (`python -m gpusci.cli loop`). Every hypothesis is logged with its claim, its pre-registered prediction, the change tested and the decision, accepted or not.

**Protocol.** Search origins 2026-08-10..2026-10-05, validation 2026-10-06..2026-10-15, locked exam 2026-10-23..2026-10-30 (never used for a decision). Horizons 1, 3, 7 days. Accept when search skill gains at least 0.003, validation does not fall and the all-series guard does not fall by more than 0.005.

**Scored universe.** 37 headline posted-rental series of the catalog at the event cutoff. **Guard universe.** 592 posted-rental series.

**Starting point:** the starter kit's drift baseline. Search skill +0.0883 (coverage 0.62), validation +0.2084.

### Iteration 1: H01 Pin the median to today's price

- **Claim:** Posted prices are sticky (menu costs): on most days nothing changes, so the best point forecast of an h-day change is zero; drift's empirical median adds noise when history is short.
- **Pre-registered prediction:** Small gain in pooled skill at h=1 and h=3; no loss at h=7.
- **Change tested:** `{"median_mode": "pinned"}` (proposed by: agent)
- **Search** pooled skill: champion +0.0883 -> candidate +0.0903; coverage 0.62 -> 0.62
- **Decision:** rejected: rejected at screening: search skill change +0.0020 < +0.003

### Iteration 2: H02 Borrow strength from peer series of the same instrument class

- **Claim:** Change distributions are similar within an instrument class (rate cards behave like rate cards). The scored rate-card series have only 60-89 days of history, too few changes to estimate a 10% tail; shrinking towards the class distribution (empirical Bayes) fixes the tails.
- **Pre-registered prediction:** Gain concentrated in the short fastgpu/gpurentalprices series; coverage closer to 0.8.
- **Change tested:** `{"peer_group": "class", "peer_n0": 20.0}` (proposed by: agent)
- **Search** pooled skill: champion +0.0883 -> candidate +0.1284; coverage 0.62 -> 0.64
- **Validation** pooled skill: champion +0.2084 -> candidate +0.2449; bootstrap 90% CI of the gain [+0.0288, +0.0456]
- **Guard** (all 592 rental series): +0.0945 -> +0.1391
- **Decision:** ACCEPTED: accepted: search +0.0401, validation +0.0364 (bootstrap P(better) 1.00), guard +0.0446

### Iteration 3: H03 Recent changes matter more (regime drift)

- **Claim:** Market conditions change over the year (new generations, demand waves); a 30-day half-life on past changes tracks the current regime.
- **Pre-registered prediction:** Small gain; larger at h=7.
- **Change tested:** `{"halflife": 30.0}` (proposed by: agent)
- **Search** pooled skill: champion +0.1284 -> candidate +0.1491; coverage 0.64 -> 0.68
- **Validation** pooled skill: champion +0.2449 -> candidate +0.2467; bootstrap 90% CI of the gain [-0.0072, +0.0132]
- **Guard** (all 592 rental series): +0.1391 -> +0.1473
- **Decision:** ACCEPTED: accepted: search +0.0207, validation +0.0018 (bootstrap P(better) 0.64), guard +0.0082

### Iteration 4: H04 Composition jumps: a provider just joined or left

- **Claim:** The daily median is an index with entry/exit bias. On days the provider count changes the median moves 90% of the time versus 12% otherwise (pack data). The state 'provider count just changed' should select a different (wider, more active) change distribution.
- **Pre-registered prediction:** Gain in the few cells that follow a provider-count change; small pooled gain.
- **Change tested:** `{"cond_comp": 10.0}` (proposed by: agent)
- **Search** pooled skill: champion +0.1491 -> candidate +0.1426; coverage 0.68 -> 0.70
- **Decision:** rejected: rejected at screening: search skill change -0.0066 < +0.003

### Iteration 5: H05 Composition jumps partly reverse

- **Claim:** If a provider drops out for a day (a scraping miss or a sold-out listing) and comes back, a median move caused by entry/exit is transient, unlike a genuine repricing. So after a provider-count jump the median forecast should move half-way back to the pre-jump price.
- **Pre-registered prediction:** Gain if transient exits are common; loss if exits are permanent.
- **Change tested:** `{"revert_comp": 0.5}` (proposed by: agent)
- **Search** pooled skill: champion +0.1491 -> candidate +0.1572; coverage 0.68 -> 0.68
- **Validation** pooled skill: champion +0.2467 -> candidate +0.2523; bootstrap 90% CI of the gain [+0.0006, +0.0117]
- **Guard** (all 592 rental series): +0.1473 -> +0.1581
- **Decision:** ACCEPTED: accepted: search +0.0080, validation +0.0056 (bootstrap P(better) 0.98), guard +0.0108

### Iteration 6: H06 Moves cluster (a price that just moved moves again)

- **Claim:** Repricing comes in bursts (a competitor cuts, others follow; marketplace books churn), so the state 'moved yesterday' predicts more movement.
- **Pre-registered prediction:** Gain at h=1 mainly.
- **Change tested:** `{"cond_jump": 5.0}` (proposed by: agent)
- **Search** pooled skill: champion +0.1572 -> candidate +0.1592; coverage 0.68 -> 0.70
- **Decision:** rejected: rejected at screening: search skill change +0.0020 < +0.003

### Iteration 7: H07 State-dependent, not Calvo, repricing (duration dependence)

- **Claim:** Under Calvo pricing the chance of a price change does not depend on how long the price has been fixed, and drift (which ignores staleness) is already right. Under state-dependent pricing the hazard varies with time since the last change. Conditioning on the staleness bucket tests this.
- **Pre-registered prediction:** If Calvo holds: no gain (a clean negative result). If state-dependent: gain.
- **Change tested:** `{"cond_stale": 5.0}` (proposed by: agent)
- **Search** pooled skill: champion +0.1572 -> candidate +0.1536; coverage 0.68 -> 0.70
- **Decision:** rejected: rejected at screening: search skill change -0.0036 < +0.003

### Iteration 8: H08 Intervals are too wide (calibration)

- **Claim:** Drift's backtest coverage is about 0.75-0.76 inside [q10, q90], near the 0.8 target, but pinball loss also rewards sharpness; a narrower interval may trade a little coverage for lower loss.
- **Pre-registered prediction:** Small gain if the tails are over-dispersed; loss otherwise.
- **Change tested:** `{"width_scale": 0.8}` (proposed by: agent)
- **Search** pooled skill: champion +0.1572 -> candidate +0.1467; coverage 0.68 -> 0.64
- **Decision:** rejected: rejected at screening: search skill change -0.0105 < +0.003

### Iteration 9: H09 Intervals are too narrow (calibration)

- **Claim:** The opposite of H08: short histories under-estimate tail risk.
- **Pre-registered prediction:** Gain only if H08 failed and coverage is well below 0.8.
- **Change tested:** `{"width_scale": 1.2}` (proposed by: agent)
- **Search** pooled skill: champion +0.1572 -> candidate +0.1589; coverage 0.68 -> 0.71
- **Decision:** rejected: rejected at screening: search skill change +0.0017 < +0.003

### Iteration 10: H10 Peers should share the collector, not just the class

- **Claim:** Each collector has its own sampling quirks (snapshot times, provider lists), so a peer group of the same class AND the same collector is more similar than the class alone.
- **Pre-registered prediction:** Gain over class-only pooling if collector quirks dominate.
- **Change tested:** `{"peer_group": "class_source"}` (proposed by: agent)
- **Search** pooled skill: champion +0.1572 -> candidate +0.1557; coverage 0.68 -> 0.69
- **Decision:** rejected: rejected at screening: search skill change -0.0015 < +0.003

### Iteration 11: H11 Stronger shrinkage towards peers

- **Claim:** With 60-89 days of history even own changes are noisy; more weight on the pooled distribution helps.
- **Pre-registered prediction:** Gain if H02 was accepted and short series still dominate the loss.
- **Change tested:** `{"peer_n0": 60.0}` (proposed by: agent)
- **Search** pooled skill: champion +0.1572 -> candidate +0.1432; coverage 0.68 -> 0.65
- **Decision:** rejected: rejected at screening: search skill change -0.0140 < +0.003

### Iteration 12: H12 Shorter memory (15-day half-life)

- **Claim:** If H03 helped, the regime may turn over even faster.
- **Pre-registered prediction:** Gain only if H03 was accepted.
- **Change tested:** `{"halflife": 15.0}` (proposed by: agent)
- **Search** pooled skill: champion +0.1572 -> candidate +0.1633; coverage 0.68 -> 0.69
- **Validation** pooled skill: champion +0.2523 -> candidate +0.2594; bootstrap 90% CI of the gain [+0.0032, +0.0109]
- **Guard** (all 592 rental series): +0.1581 -> +0.1567
- **Decision:** ACCEPTED: accepted: search +0.0061, validation +0.0072 (bootstrap P(better) 1.00), guard -0.0014

## Locked exam (scored once, after the loop)

| model | search_skill | validation_skill | exam_skill | exam_skill_h1 | exam_skill_h3 | exam_skill_h7 | exam_median_of_series | exam_geomean | exam_coverage | exam_guard_skill |
|---|---|---|---|---|---|---|---|---|---|---|
| last_value (flat) | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.3926 | +0.0000 |
| drift (starter) | +0.0883 | +0.2084 | +0.0698 | +0.0852 | +0.1117 | +0.0421 | +0.0870 | +0.0689 | +0.7210 | +0.0939 |
| champion after start | +0.0883 | +0.2084 | +0.0698 | +0.0852 | +0.1117 | +0.0421 | +0.0870 | +0.0689 | +0.7210 | +0.0939 |
| champion after H02 | +0.1284 | +0.2449 | +0.1202 | +0.0829 | +0.1672 | +0.1049 | +0.1122 | +0.1098 | +0.6938 | +0.1589 |
| champion after H03 | +0.1491 | +0.2467 | +0.1267 | +0.0803 | +0.1576 | +0.1229 | +0.1017 | +0.1086 | +0.7481 | +0.1743 |
| champion after H05 | +0.1572 | +0.2523 | +0.1107 | +0.0727 | +0.1281 | +0.1121 | +0.0816 | +0.0961 | +0.7481 | +0.1759 |
| champion after H12 | +0.1633 | +0.2594 | +0.1091 | +0.0792 | +0.1335 | +0.1041 | +0.0715 | +0.0860 | +0.7519 | +0.1808 |

Champion minus drift on the exam, pooled skill: +0.0393, 90% interval [+0.0163, +0.0655] by resampling whole origins (8 origins); P(champion better) = 1.00.
