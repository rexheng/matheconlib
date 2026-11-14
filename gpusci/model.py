"""The quantile forecaster.

Idea: the forecast for series s at origin t and horizon h is today's price times exp(r), where r
is drawn from a *weighted empirical distribution of past h-day log changes*. Pinball loss is
minimised by the true quantiles of the predictive distribution, so we report its 0.1, 0.5 and
0.9 quantiles. Because most posted prices do not move, that distribution has a big atom at r = 0,
and its quantiles collapse onto today's price unless a move in one direction is likely enough.

Every research hypothesis is a change to `Config`: which past changes count (own series only, or
pooled with peer series), how much recent changes weigh, whether the state of the market today
(a provider just joined or left, the price just moved, how long since the last move) selects
which past changes are relevant, and whether the median is pinned to the last value.

`Config()` with no arguments reproduces the starter kit's `drift` baseline (up to the quantile
interpolation rule).

Leakage rule, enforced here: at origin t only records whose *end* day is <= t are used, and the
state of the market at the origin is read from observations dated <= t.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import math

import numpy as np

from .data import Series

TAUS = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class Config:
    # which past changes form the predictive distribution
    halflife: float | None = None        # recency weighting of past changes, days; None = equal weights
    lookback: int | None = None          # ignore changes that ended more than this many days ago
    peer_group: str | None = None        # None | "class" | "class_source" | "all": pool changes of peer series
    peer_n0: float = 20.0                # shrinkage strength: peers get weight n0 / (n0 + n_own)
    min_records: int = 20                # fewer usable changes than this -> flat last value (starter rule)
    # conditioning on the state of the market at the origin
    cond_comp: float = 1.0               # weight multiplier for past changes whose start state matches
                                         # today's "provider count just changed" flag (1 = off)
    cond_jump: float = 1.0               # same for "price moved on the last day"
    cond_stale: float = 1.0              # same for staleness bucket (days since last move)
    # shape of the output
    median_mode: str = "empirical"       # "empirical" | "pinned" (q50 = last value)
    width_scale: float = 1.0             # multiply the log-distance of q10 and q90 from the last value
    flat_classes: tuple = ()             # instrument classes forced to a flat last-value forecast
    revert_comp: float = 0.0             # after a provider-count jump, shift q50 back toward the
                                         # pre-jump price by this fraction (0 = off)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["flat_classes"] = list(self.flat_classes)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Config":
        d = dict(d)
        d["flat_classes"] = tuple(d.get("flat_classes", ()))
        return Config(**d)

    def with_(self, **kw) -> "Config":
        if "flat_classes" in kw:
            kw["flat_classes"] = tuple(kw["flat_classes"])
        return replace(self, **kw)


# ----------------------------------------------------------------------------------------------
# Records: every past h-day change of a series, with the market state on its start day.

STALE_EDGES = (1, 3, 7, 14, 30)  # buckets of days since the last price move


def _stale_bucket(days_since: float) -> int:
    for k, e in enumerate(STALE_EDGES):
        if days_since <= e:
            return k
    return len(STALE_EDGES)


def _state_arrays(s: Series):
    """Per observation index i, the state known on day days[i] (uses only indices <= i)."""
    n = len(s.days)
    comp = np.zeros(n, dtype=bool)
    jump = np.zeros(n, dtype=bool)
    stale = np.zeros(n, dtype=np.int64)
    prev_val = np.full(n, np.nan)
    last_move_day = s.days[0] if n else 0
    for i in range(n):
        if i > 0 and s.days[i] - s.days[i - 1] == 1:
            jump[i] = s.vals[i] != s.vals[i - 1]
            a, b = s.nprov[i], s.nprov[i - 1]
            comp[i] = (not (math.isnan(a) or math.isnan(b))) and a != b
            prev_val[i] = s.vals[i - 1]
        if i > 0 and s.vals[i] != s.vals[i - 1]:
            last_move_day = s.days[i]
        stale[i] = _stale_bucket(s.days[i] - last_move_day if i > 0 else 999)
    return comp, jump, stale, prev_val


def records(s: Series, h: int) -> dict:
    """All h-day log changes (start index i, end index j with days[j] = days[i] + h)."""
    if h in s._records:
        return s._records[h]
    if "_state" not in s._records:
        s._records["_state"] = _state_arrays(s)
    comp, jump, stale, _ = s._records["_state"]
    target = s.days + h
    j = np.searchsorted(s.days, target)
    ok = (j < len(s.days))
    ok[ok] &= s.days[j[ok]] == target[ok]
    i_idx = np.nonzero(ok)[0]
    j_idx = j[ok]
    rec = {
        "end": s.days[j_idx],
        "r": np.log(s.vals[j_idx] / s.vals[i_idx]),
        "comp": comp[i_idx],
        "jump": jump[i_idx],
        "stale": stale[i_idx],
    }
    s._records[h] = rec
    return rec


def origin_state(s: Series, k: int):
    comp, jump, stale, prev_val = s._records.get("_state") or _state_arrays(s)
    if "_state" not in s._records:
        s._records["_state"] = (comp, jump, stale, prev_val)
    return bool(comp[k]), bool(jump[k]), int(stale[k]), float(prev_val[k])


# ----------------------------------------------------------------------------------------------

def _weights(rec: dict, upto: int, t: int, cfg: Config, state) -> tuple[np.ndarray, np.ndarray]:
    end = rec["end"][:upto]
    w = np.ones(upto)
    if cfg.halflife:
        w *= 0.5 ** ((t - end) / cfg.halflife)
    if cfg.lookback:
        w *= (t - end) <= cfg.lookback
    comp, jump, stale = state
    if cfg.cond_comp != 1.0:
        w *= np.where(rec["comp"][:upto] == comp, cfg.cond_comp, 1.0)
    if cfg.cond_jump != 1.0:
        w *= np.where(rec["jump"][:upto] == jump, cfg.cond_jump, 1.0)
    if cfg.cond_stale != 1.0:
        w *= np.where(rec["stale"][:upto] == stale, cfg.cond_stale, 1.0)
    return rec["r"][:upto], w


def weighted_quantiles(x: np.ndarray, w: np.ndarray, taus=TAUS) -> np.ndarray:
    """Inverted-CDF weighted quantiles: the smallest x whose cumulative weight share >= tau.
    Keeps atoms intact (a price that usually does not move gives exactly 0)."""
    o = np.argsort(x, kind="mergesort")
    xs, ws = x[o], w[o]
    c = np.cumsum(ws)
    c /= c[-1]
    idx = np.searchsorted(c, np.asarray(taus) - 1e-12, side="left")
    return xs[np.minimum(idx, len(xs) - 1)]


def group_key(s: Series, how: str | None):
    if how is None:
        return None
    if how == "class":
        return s.instrument_class
    if how == "class_source":
        return (s.instrument_class, s.source_id)
    if how == "all":
        return "all"
    raise ValueError(how)


class PeerPools:
    """Pooled change records per (group, h), compressed per origin to a quantile grid."""

    GRID = 400

    def __init__(self, store: dict[str, Series], how: str | None):
        self.how = how
        self.members: dict = {}
        for s in store.values():
            self.members.setdefault(group_key(s, how), []).append(s)
        self._cat: dict = {}
        self._cache: dict = {}

    def _concat(self, key, h):
        ck = (key, h)
        if ck not in self._cat:
            parts = [records(s, h) for s in self.members.get(key, [])]
            parts = [p for p in parts if len(p["r"])]
            if not parts:
                self._cat[ck] = None
            else:
                cat = {f: np.concatenate([p[f] for p in parts]) for f in parts[0]}
                o = np.argsort(cat["end"], kind="mergesort")
                self._cat[ck] = {f: v[o] for f, v in cat.items()}
        return self._cat[ck]

    def grid(self, key, h, t, cfg: Config, state):
        ck = (key, h, t, state, cfg)
        if ck in self._cache:
            return self._cache[ck]
        rec = self._concat(key, h)
        out = None
        if rec is not None:
            upto = int(np.searchsorted(rec["end"], t, side="right"))
            if upto >= cfg.min_records:
                r, w = _weights(rec, upto, t, cfg, state)
                if w.sum() > 0:
                    taus = (np.arange(self.GRID) + 0.5) / self.GRID
                    out = (weighted_quantiles(r, w, taus), upto)
        self._cache[ck] = out
        return out


def forecast(s: Series, t: int, h: int, cfg: Config, pools: PeerPools | None):
    """Return (q10, q50, q90, last_value, note) for series s at origin day t, horizon h days,
    or None when the series has no observation on or before t."""
    k = s.last_index_on_or_before(t)
    if k < 0:
        return None
    L = float(s.vals[k])
    if s.instrument_class in cfg.flat_classes:
        return L, L, L, L, "flat_class"
    h_eff = int(t + h - s.days[k])   # the forecast spans from the last observation to the target
    rec = records(s, h_eff)
    comp, jump, stale, prev_val = origin_state(s, k)
    state = (comp, jump, stale)
    upto = int(np.searchsorted(rec["end"], t, side="right"))
    r_own, w_own = _weights(rec, upto, t, cfg, state)
    n_own = float(w_own.sum()) if upto else 0.0

    peer = None
    if pools is not None and cfg.peer_group is not None:
        peer = pools.grid(group_key(s, cfg.peer_group), h_eff, t, cfg, state)

    if peer is None:
        if upto < cfg.min_records or n_own <= 0:
            return L, L, L, L, "flat_fewer_than_min_records"
        Q = weighted_quantiles(r_own, w_own)
    else:
        grid, _ = peer
        alpha = cfg.peer_n0 / (cfg.peer_n0 + n_own) if n_own > 0 else 1.0
        if upto > 0 and n_own > 0:
            x = np.concatenate([r_own, grid])
            w = np.concatenate([(1 - alpha) * w_own / n_own, np.full(len(grid), alpha / len(grid))])
        else:
            x, w = grid, np.full(len(grid), 1.0 / len(grid))
        Q = weighted_quantiles(x, w)

    q10, q50, q90 = (float(v) for v in Q)
    if cfg.median_mode == "pinned":
        q50 = 0.0
    if cfg.revert_comp and comp and not math.isnan(prev_val) and prev_val > 0:
        q50 += cfg.revert_comp * math.log(prev_val / L)
    q10 *= cfg.width_scale
    q90 *= cfg.width_scale
    lo, mid, hi = sorted((q10, q50, q90))
    lo, mid, hi = (L * math.exp(v) for v in (lo, mid, hi))
    lo = max(lo, 1e-6)
    return lo, mid, hi, L, "model"
