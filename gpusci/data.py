"""Load the starter kit's series file (series_daily.csv / history_<cutoff>.csv) and catalog.

Every series becomes a few numpy arrays indexed by integer day (days since 1970-01-01), so the
model can cut history at any origin with a binary search. Nothing here looks past an origin:
callers pass the origin and the model only reads indices whose day is <= origin.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPOCH = np.datetime64("1970-01-01", "D")


def to_day(x) -> np.ndarray | int:
    """ISO date(s) -> integer day number."""
    if isinstance(x, str):
        return int((np.datetime64(x, "D") - EPOCH).astype(int))
    arr = pd.to_datetime(pd.Series(x)).values.astype("datetime64[D]")
    return (arr - EPOCH).astype(int)


def to_iso(day: int) -> str:
    return str(EPOCH + np.timedelta64(int(day), "D"))


@dataclass
class Series:
    sid: str
    days: np.ndarray          # int, sorted ascending
    vals: np.ndarray          # float, USD per GPU-hour
    nprov: np.ndarray         # float (nan for AWS composites)
    instrument_class: str
    source_id: str
    gpu_model: str
    pricing_tier: str
    tier: str                 # headline / extended in the catalog used
    _records: dict = field(default_factory=dict, repr=False)

    def last_index_on_or_before(self, day: int) -> int:
        """Index of the last observation dated <= day, or -1."""
        return int(np.searchsorted(self.days, day, side="right")) - 1

    def value_on(self, day: int) -> float | None:
        i = int(np.searchsorted(self.days, day))
        if i < len(self.days) and self.days[i] == day:
            return float(self.vals[i])
        return None


def load_store(series_csv: str, catalog_csv: str, include_aws: bool = False) -> dict[str, Series]:
    cols = ["series_id", "date", "value", "n_providers"]
    df = pd.read_csv(series_csv, usecols=cols)
    if not include_aws:
        df = df[~df.series_id.str.startswith("aws_spot.")]
    cat = pd.read_csv(catalog_csv).set_index("series_id")
    df["day"] = to_day(df["date"])
    df = df.sort_values(["series_id", "day"])
    store: dict[str, Series] = {}
    for sid, g in df.groupby("series_id", sort=False):
        if sid not in cat.index:
            continue
        c = cat.loc[sid]
        store[sid] = Series(
            sid=sid,
            days=g["day"].to_numpy(dtype=np.int64),
            vals=g["value"].to_numpy(dtype=float),
            nprov=g["n_providers"].to_numpy(dtype=float),
            instrument_class=str(c["instrument_class"]),
            source_id=str(c["source_id"]),
            gpu_model=str(c["gpu_model"]),
            pricing_tier=str(c["pricing_tier"]),
            tier=str(c["tier"]),
        )
    return store


def headline_ids(catalog_csv: str, include_aws: bool = False) -> list[str]:
    cat = pd.read_csv(catalog_csv)
    m = cat.tier == "headline"
    if not include_aws:
        m &= ~cat.series_id.str.startswith("aws_spot.")
    return sorted(cat.loc[m, "series_id"])
