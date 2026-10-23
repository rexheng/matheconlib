"""Microeconomic simulations.

Posted-price monopoly, Becker's cutoff theory of crime, matching pennies,
Cournot best-response dynamics, and Gale-Shapley deferred acceptance.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def uniform_optimal_price(lower: float, upper: float, continuation: float) -> float:
    """Monopoly posted price for values uniform on ``[lower, upper]``.

    This is `ValueDistribution.uniformOptimalPrice`: the unconstrained peak
    ``(upper + continuation) / 2``, projected onto the support. With
    ``lower = 0``, ``upper = 1`` and zero continuation value the price is
    one half, which `MathEconLibTest/ValueDistribution/Examples.lean` checks.
    """
    if not upper > lower:
        raise ValueError("upper endpoint must be strictly above lower")
    peak = (upper + continuation) / 2.0
    return float(max(lower, min(upper, peak)))


def uniform_seller_profit(
    price: float, lower: float, upper: float, continuation: float
) -> float:
    """Expected profit ``(price - continuation) * P(value >= price)``.

    Demand is the survival function of ``Uniform[lower, upper]``. A price
    outside the support sells to everyone or to no one.
    """
    if not upper > lower:
        raise ValueError("upper endpoint must be strictly above lower")
    if price <= lower:
        demand = 1.0
    elif price >= upper:
        demand = 0.0
    else:
        demand = (upper - price) / (upper - lower)
    return float((price - continuation) * demand)


def simulate_posted_price_revenue(
    price: float,
    lower: float,
    upper: float,
    continuation: float,
    n: int = 20_000,
    seed: int = 0,
) -> float:
    """Sample mean of per-buyer profit at a posted price."""
    values = np.random.default_rng(seed).uniform(lower, upper, size=n)
    return float(np.mean((price - continuation) * (values >= price)))


def becker_crime_rate(gains: Array, probability: float, fine: float) -> float:
    """Share of agents who offend.

    An agent with net gain ``v`` commits the offense when ``v >= p * f``,
    and abstains otherwise. Ties commit, matching
    `crimeProblem_strategy_commit_iff` in
    `MathEconLibExamples/SingleParameter/BeckerCrime.lean`.
    """
    sanction = probability * fine
    return float(np.mean(np.asarray(gains, dtype=float) >= sanction))


def becker_crime_rate_uniform(
    probability: float, fine: float, low: float = 0.0, high: float = 1.0
) -> float:
    """Closed-form crime rate when net gains are uniform on ``[low, high]``."""
    if not high > low:
        raise ValueError("high must be strictly above low")
    sanction = probability * fine
    if sanction <= low:
        return 1.0
    if sanction >= high:
        return 0.0
    return float((high - sanction) / (high - low))


def fictitious_play_matching_pennies(rounds: int, seed: int = 0) -> Array:
    """Empirical frequency of heads for each player under fictitious play.

    Player 0 wins when the coins match. Player 1 wins when they differ.
    Each player best-responds to the opponent's empirical frequency so far.
    Exact indifference is broken by a fair coin. The unique mixed Nash,
    `matchingPennies_uniform_is_mixed_nash`, puts probability one half on
    each action. Returned array has shape ``(rounds, 2)``.
    """
    if rounds < 1:
        raise ValueError("rounds must be positive")
    rng = np.random.default_rng(seed)
    heads_seen = np.zeros(2, dtype=float)
    plays = np.empty((rounds, 2), dtype=np.int8)
    for t in range(rounds):
        beliefs = np.full(2, 0.5) if t == 0 else heads_seen / t
        for player in (0, 1):
            p_heads = beliefs[1 - player]
            # Payoff gap of heads over tails. Positive means heads is better.
            gap = (2.0 * p_heads - 1.0) if player == 0 else (1.0 - 2.0 * p_heads)
            if gap > 0:
                action = 0
            elif gap < 0:
                action = 1
            else:
                action = int(rng.integers(0, 2))
            plays[t, player] = action
        heads_seen += plays[t] == 0
    counts = np.cumsum(plays == 0, axis=0)
    return counts / np.arange(1, rounds + 1, dtype=float)[:, None]


def cournot_nash(intercept: float, slope: float, cost: float, firms: int = 2) -> float:
    """Symmetric Cournot quantity per firm.

    Inverse demand is ``P = intercept - slope * total quantity``. Every firm
    has constant marginal cost ``cost``.
    """
    if firms < 1:
        raise ValueError("firms must be positive")
    if slope <= 0:
        raise ValueError("slope must be positive")
    if intercept <= cost:
        return 0.0
    return float((intercept - cost) / (slope * (firms + 1)))


def cournot_best_response(
    other_quantity: float, intercept: float, slope: float, cost: float
) -> float:
    """One firm's best response to the combined output of the others."""
    quantity = (intercept - cost - slope * other_quantity) / (2.0 * slope)
    return float(max(0.0, quantity))


def cournot_dynamics(
    initial: Array,
    intercept: float,
    slope: float,
    cost: float,
    rounds: int = 40,
) -> Array:
    """Simultaneous best-response dynamics. Row ``t`` is the quantity vector."""
    quantities = np.empty((rounds, len(initial)), dtype=float)
    current = np.asarray(initial, dtype=float).copy()
    quantities[0] = current
    for t in range(1, rounds):
        total = float(current.sum())
        nxt = np.array(
            [
                cournot_best_response(total - q, intercept, slope, cost)
                for q in current
            ]
        )
        quantities[t] = nxt
        current = nxt
    return quantities


def deferred_acceptance(
    proposer_prefs: dict[str, list[str]],
    reviewer_prefs: dict[str, list[str]],
) -> dict[str, str | None]:
    """Proposer-side Gale-Shapley deferred acceptance.

    Preferences are lists from best to worst, and every listed partner is
    acceptable. Someone omitted from a list is unacceptable. The result maps
    each proposer to a match, or ``None`` if they stay single.

    `MarriageMarket.galeShapleyFinset` is the executable finite version of
    this algorithm in the Lean library. This function is a numerical twin,
    not a proof.
    """
    rank = {
        reviewer: {proposer: i for i, proposer in enumerate(prefs)}
        for reviewer, prefs in reviewer_prefs.items()
    }
    next_choice = {proposer: 0 for proposer in proposer_prefs}
    partner: dict[str, str | None] = {proposer: None for proposer in proposer_prefs}
    held: dict[str, str] = {}
    free = list(proposer_prefs)
    while free:
        proposer = free.pop(0)
        prefs = proposer_prefs[proposer]
        cursor = next_choice[proposer]
        if cursor >= len(prefs):
            continue
        reviewer = prefs[cursor]
        next_choice[proposer] = cursor + 1
        if reviewer not in rank or proposer not in rank[reviewer]:
            free.append(proposer)
            continue
        incumbent = held.get(reviewer)
        if incumbent is None or rank[reviewer][proposer] < rank[reviewer][incumbent]:
            if incumbent is not None:
                partner[incumbent] = None
                free.append(incumbent)
            held[reviewer] = proposer
            partner[proposer] = reviewer
        else:
            free.append(proposer)
    return partner


def is_stable(
    matching: dict[str, str | None],
    proposer_prefs: dict[str, list[str]],
    reviewer_prefs: dict[str, list[str]],
) -> bool:
    """No blocking pair, treating omission from a preference list as unacceptable."""
    reviewer_partner = {
        reviewer: proposer
        for proposer, reviewer in matching.items()
        if reviewer is not None
    }
    proposer_rank = {
        proposer: {reviewer: i for i, reviewer in enumerate(prefs)}
        for proposer, prefs in proposer_prefs.items()
    }
    reviewer_rank = {
        reviewer: {proposer: i for i, proposer in enumerate(prefs)}
        for reviewer, prefs in reviewer_prefs.items()
    }
    for proposer, prefs in proposer_prefs.items():
        current = matching.get(proposer)
        for reviewer in prefs:
            if current is not None and proposer_rank[proposer][reviewer] >= proposer_rank[proposer][current]:
                break
            if reviewer not in reviewer_rank or proposer not in reviewer_rank[reviewer]:
                continue
            incumbent = reviewer_partner.get(reviewer)
            if incumbent is None or reviewer_rank[reviewer][proposer] < reviewer_rank[reviewer][incumbent]:
                return False
    return True


# Two applicants and two departments. Applicant-proposing and
# department-proposing deferred acceptance produce different stable matches:
# each side gets its favorite when it is the one that proposes.
APPLICANTS = {
    "Ada": ["Theory", "Metrics"],
    "Grace": ["Metrics", "Theory"],
}
DEPARTMENTS = {
    "Theory": ["Grace", "Ada"],
    "Metrics": ["Ada", "Grace"],
}
