"""Pinball loss and skill, matching the starter kit's scorer.py definitions.

Cell = (series, origin, horizon). Cell loss = mean over q in (0.1, 0.5, 0.9) of
max(q (y - f), (q - 1)(y - f)). Baseline = the flat last value on or before the origin, whose
cell loss is 0.5 |y - last|. Pooled skill = 1 - sum(loss) / sum(baseline loss) over scored cells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import TAUS


def pinball(y: np.ndarray, q10, q50, q90) -> np.ndarray:
    tot = 0.0
    for tau, f in zip(TAUS, (q10, q50, q90)):
        d = y - f
        tot = tot + np.maximum(tau * d, (tau - 1) * d)
    return tot / len(TAUS)


def add_losses(cells: pd.DataFrame) -> pd.DataFrame:
    c = cells.copy()
    c["loss"] = pinball(c.y.values, c.q10.values, c.q50.values, c.q90.values)
    c["base_loss"] = 0.5 * np.abs(c.y.values - c["last"].values)
    return c


def summarize(cells: pd.DataFrame) -> dict:
    """Aggregates over scored cells (cells must carry loss, base_loss, y, q10, q90, series_id)."""
    if len(cells) == 0:
        return {"cells": 0}
    L, B = cells.loss.sum(), cells.base_loss.sum()
    per = cells.groupby("series_id").agg(loss=("loss", "mean"), base=("base_loss", "mean"))
    per = per[per.base > 0]
    ratio = (per.loss / per.base).clip(lower=1e-3)
    inside = ((cells.y >= cells.q10) & (cells.y <= cells.q90)).mean()
    below = (cells.y < cells.q10).mean()
    above = (cells.y > cells.q90).mean()
    out = {
        "cells": int(len(cells)),
        "series": int(cells.series_id.nunique()),
        "loss": float(cells.loss.mean()),
        "base_loss": float(cells.base_loss.mean()),
        "skill_pooled": float(1 - L / B) if B > 0 else None,
        "skill_median_of_series": float(np.median(1 - ratio)) if len(ratio) else None,
        "skill_geomean": float(1 - np.exp(np.log(ratio).mean())) if len(ratio) else None,
        "share_series_positive": float((ratio < 1).mean()) if len(ratio) else None,
        "coverage_inside": float(inside),
        "coverage_below": float(below),
        "coverage_above": float(above),
    }
    for h, g in cells.groupby("horizon_days"):
        b = g.base_loss.sum()
        out[f"skill_pooled_h{int(h)}"] = float(1 - g.loss.sum() / b) if b > 0 else None
    return out


def block_bootstrap_diff(cells_a: pd.DataFrame, cells_b: pd.DataFrame, n: int = 2000, seed: int = 0) -> dict:
    """Paired comparison of two methods on the same cells, resampling whole origins (cells within an
    origin are correlated). Returns the skill difference (a - b) and a 90% interval, plus the share
    of resamples where a beats b. A Diebold-Mariano-style test that respects cross-series correlation."""
    key = ["series_id", "origin_date", "horizon_days"]
    m = cells_a[key + ["loss", "base_loss"]].merge(cells_b[key + ["loss"]], on=key, suffixes=("_a", "_b"))
    if len(m) == 0:
        return {}
    g = m.groupby("origin_date").agg(la=("loss_a", "sum"), lb=("loss_b", "sum"), bb=("base_loss", "sum"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    la, lb, bb = g.la.values[idx].sum(1), g.lb.values[idx].sum(1), g.bb.values[idx].sum(1)
    ok = bb > 0
    diff = (lb[ok] - la[ok]) / bb[ok]   # = skill_a - skill_b
    point = (g.lb.sum() - g.la.sum()) / g.bb.sum()
    return {
        "skill_diff": float(point),
        "ci90": [float(np.quantile(diff, 0.05)), float(np.quantile(diff, 0.95))],
        "p_a_better": float((diff > 0).mean()),
        "origins": int(len(g)),
    }
