"""The research agenda: each hypothesis is an economic claim, a pre-registered prediction and the
change to the forecaster that tests it. Written before any of them was run on the validation or
exam origins. The loop (scientist.py) tests them in this order, greedily, against the current
champion, and logs every outcome, including rejections.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Hypothesis:
    hid: str
    title: str
    claim: str            # the economic / statistical claim
    prediction: str       # what we expect to see if the claim is true (pre-registered)
    delta: dict           # change to the champion Config
    proposer: str = "agent"   # who proposed it: "agent" = the AI agent (Claude) that wrote this agenda
    tags: list = field(default_factory=list)


AGENDA: list[Hypothesis] = [
    Hypothesis(
        "H01", "Pin the median to today's price",
        "Posted prices are sticky (menu costs): on most days nothing changes, so the best point "
        "forecast of an h-day change is zero; drift's empirical median adds noise when history is short.",
        "Small gain in pooled skill at h=1 and h=3; no loss at h=7.",
        {"median_mode": "pinned"}, tags=["stickiness"]),
    Hypothesis(
        "H02", "Borrow strength from peer series of the same instrument class",
        "Change distributions are similar within an instrument class (rate cards behave like rate cards). "
        "The scored rate-card series have only 60-89 days of history, too few changes to estimate a 10% tail; "
        "shrinking towards the class distribution (empirical Bayes) fixes the tails.",
        "Gain concentrated in the short fastgpu/gpurentalprices series; coverage closer to 0.8.",
        {"peer_group": "class", "peer_n0": 20.0}, tags=["pooling"]),
    Hypothesis(
        "H03", "Recent changes matter more (regime drift)",
        "Market conditions change over the year (new generations, demand waves); a 30-day half-life "
        "on past changes tracks the current regime.",
        "Small gain; larger at h=7.",
        {"halflife": 30.0}, tags=["recency"]),
    Hypothesis(
        "H04", "Composition jumps: a provider just joined or left",
        "The daily median is an index with entry/exit bias. On days the provider count changes the median "
        "moves 90% of the time versus 12% otherwise (pack data). The state 'provider count just changed' "
        "should select a different (wider, more active) change distribution.",
        "Gain in the few cells that follow a provider-count change; small pooled gain.",
        {"cond_comp": 10.0}, tags=["composition"]),
    Hypothesis(
        "H05", "Composition jumps partly reverse",
        "If a provider drops out for a day (a scraping miss or a sold-out listing) and comes back, a median "
        "move caused by entry/exit is transient, unlike a genuine repricing. So after a provider-count jump "
        "the median forecast should move half-way back to the pre-jump price.",
        "Gain if transient exits are common; loss if exits are permanent.",
        {"revert_comp": 0.5}, tags=["composition"]),
    Hypothesis(
        "H06", "Moves cluster (a price that just moved moves again)",
        "Repricing comes in bursts (a competitor cuts, others follow; marketplace books churn), so the "
        "state 'moved yesterday' predicts more movement.",
        "Gain at h=1 mainly.",
        {"cond_jump": 5.0}, tags=["state-dependence"]),
    Hypothesis(
        "H07", "State-dependent, not Calvo, repricing (duration dependence)",
        "Under Calvo pricing the chance of a price change does not depend on how long the price has been "
        "fixed, and drift (which ignores staleness) is already right. Under state-dependent pricing the "
        "hazard varies with time since the last change. Conditioning on the staleness bucket tests this.",
        "If Calvo holds: no gain (a clean negative result). If state-dependent: gain.",
        {"cond_stale": 5.0}, tags=["state-dependence", "economics"]),
    Hypothesis(
        "H08", "Intervals are too wide (calibration)",
        "Drift's backtest coverage is about 0.75-0.76 inside [q10, q90], near the 0.8 target, but pinball "
        "loss also rewards sharpness; a narrower interval may trade a little coverage for lower loss.",
        "Small gain if the tails are over-dispersed; loss otherwise.",
        {"width_scale": 0.8}, tags=["calibration"]),
    Hypothesis(
        "H09", "Intervals are too narrow (calibration)",
        "The opposite of H08: short histories under-estimate tail risk.",
        "Gain only if H08 failed and coverage is well below 0.8.",
        {"width_scale": 1.2}, tags=["calibration"]),
    Hypothesis(
        "H10", "Peers should share the collector, not just the class",
        "Each collector has its own sampling quirks (snapshot times, provider lists), so a peer group of the "
        "same class AND the same collector is more similar than the class alone.",
        "Gain over class-only pooling if collector quirks dominate.",
        {"peer_group": "class_source"}, tags=["pooling"]),
    Hypothesis(
        "H11", "Stronger shrinkage towards peers",
        "With 60-89 days of history even own changes are noisy; more weight on the pooled distribution helps.",
        "Gain if H02 was accepted and short series still dominate the loss.",
        {"peer_n0": 60.0}, tags=["pooling"]),
    Hypothesis(
        "H12", "Shorter memory (15-day half-life)",
        "If H03 helped, the regime may turn over even faster.",
        "Gain only if H03 was accepted.",
        {"halflife": 15.0}, tags=["recency"]),
]
