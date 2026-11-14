"""Rolling-origin backtests: run a Config over origins and a universe of series, score every cell.

Universe at an origin t: the given series ids that have an observation within the last
`max_stale` days on or before t and at least `min_hist` observations on or before t.
Truth for a cell is the value dated exactly t + h; cells without it are not scored (as in the
starter kit). All truths come from inside the pack, so backtests are development feedback.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Series, to_iso
from .model import Config, PeerPools, forecast
from .score import add_losses

HORIZONS = (1, 3, 7)


def eligible(s: Series, t: int, max_stale: int = 3, min_hist: int = 7) -> bool:
    k = s.last_index_on_or_before(t)
    return k >= 0 and (t - s.days[k]) <= max_stale and (k + 1) >= min_hist


def run(store: dict[str, Series], ids: list[str], origins: list[int], cfg: Config,
        horizons=HORIZONS, pools: PeerPools | None = None, need_truth: bool = True) -> pd.DataFrame:
    if cfg.peer_group is not None and (pools is None or pools.how != cfg.peer_group):
        pools = PeerPools(store, cfg.peer_group)
    rows = []
    for t in origins:
        for sid in ids:
            s = store.get(sid)
            if s is None or not eligible(s, t):
                continue
            for h in horizons:
                y = s.value_on(t + h)
                if need_truth and y is None:
                    continue
                f = forecast(s, t, h, cfg, pools)
                if f is None:
                    continue
                q10, q50, q90, last, note = f
                rows.append((sid, t, h, q10, q50, q90, last, np.nan if y is None else y, note))
    cells = pd.DataFrame(rows, columns=["series_id", "origin", "horizon_days", "q10", "q50", "q90",
                                        "last", "y", "note"])
    cells["origin_date"] = [to_iso(t) for t in cells.origin]
    if need_truth and len(cells):
        cells = add_losses(cells)
    return cells
