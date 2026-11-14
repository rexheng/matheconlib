"""The AI-scientist loop: propose -> test -> keep or reject -> record -> repeat.

Protocol (fixed before any hypothesis was run; see PROTOCOL below):

* Origins are split in time into three disjoint sets whose target dates do not overlap:
    search      2026-08-10 .. 2026-10-05   (targets up to 2026-10-12)  used to screen hypotheses
    validation  2026-10-06 .. 2026-10-15   (targets up to 2026-10-22)  must confirm a screened gain
    exam        2026-10-23 .. 2026-10-30   (targets up to 2026-11-06)  LOCKED: never used for a decision;
                                                                      scored once, after the loop ends
* Universe "scored": the 37 headline series of the catalog at the event cutoff (the posted rental
  series most likely scored on Sunday). Universe "guard": every posted rental series in the pack
  (592), checked on every third search origin so a change that helps the headline set cannot quietly
  damage the sticky series (whose last-value loss is often zero).
* A candidate = champion + the hypothesis' change. It is accepted when
    search pooled skill improves by at least MIN_GAIN, and
    validation pooled skill does not fall, and
    guard pooled skill does not fall by more than GUARD_TOL.
* Every outcome is appended to the lab notebook (JSONL + Markdown), accepted or not.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from .backtest import run
from .data import Series, to_day, to_iso
from .hypotheses import Hypothesis
from .model import Config
from .score import block_bootstrap_diff, summarize

MIN_GAIN = 0.003
GUARD_TOL = 0.005


def day_range(a: str, b: str, step: int = 1) -> list[int]:
    return list(range(to_day(a), to_day(b) + 1, step))


PROTOCOL = {
    "search": ("2026-08-10", "2026-10-05"),
    "validation": ("2026-10-06", "2026-10-15"),
    "exam": ("2026-10-23", "2026-10-30"),
    "min_gain": MIN_GAIN,
    "guard_tol": GUARD_TOL,
    "horizons": [1, 3, 7],
}


class Lab:
    def __init__(self, store: dict[str, Series], scored_ids: list[str], guard_ids: list[str],
                 notebook_jsonl: str, notebook_md: str):
        self.store = store
        self.scored = scored_ids
        self.guard = guard_ids
        self.search = day_range(*PROTOCOL["search"])
        self.validation = day_range(*PROTOCOL["validation"])
        self.exam = day_range(*PROTOCOL["exam"])
        self.guard_origins = self.search[::3]
        self.nb_jsonl = notebook_jsonl
        self.nb_md = notebook_md
        self._cache: dict = {}

    # -- evaluation ------------------------------------------------------------------------
    def cells(self, cfg: Config, which: str) -> pd.DataFrame:
        key = (cfg, which)
        if key not in self._cache:
            if which == "search":
                c = run(self.store, self.scored, self.search, cfg)
            elif which == "validation":
                c = run(self.store, self.scored, self.validation, cfg)
            elif which == "guard":
                c = run(self.store, self.guard, self.guard_origins, cfg)
            elif which == "exam":
                c = run(self.store, self.scored, self.exam, cfg)
            elif which == "exam_guard":
                c = run(self.store, self.guard, self.exam, cfg)
            else:
                raise ValueError(which)
            self._cache[key] = c
        return self._cache[key]

    def metrics(self, cfg: Config, which: str) -> dict:
        return summarize(self.cells(cfg, which))

    # -- the loop --------------------------------------------------------------------------
    def test(self, champion: Config, hyp: Hypothesis, iteration: int) -> tuple[bool, Config, dict]:
        cand = champion.with_(**hyp.delta)
        t0 = time.time()
        if cand == champion:
            entry = self._entry(iteration, hyp, champion, cand, {}, {}, {}, {}, False,
                                "no-op: the change is already part of the champion", t0)
            return False, champion, entry
        s_c, s_n = self.metrics(champion, "search"), self.metrics(cand, "search")
        d_search = s_n["skill_pooled"] - s_c["skill_pooled"]
        if d_search < MIN_GAIN:
            entry = self._entry(iteration, hyp, champion, cand, s_c, s_n, {}, {}, False,
                                f"rejected at screening: search skill change {d_search:+.4f} < +{MIN_GAIN}", t0)
            return False, champion, entry
        v_c, v_n = self.metrics(champion, "validation"), self.metrics(cand, "validation")
        g_c, g_n = self.metrics(champion, "guard"), self.metrics(cand, "guard")
        d_val = v_n["skill_pooled"] - v_c["skill_pooled"]
        d_guard = g_n["skill_pooled"] - g_c["skill_pooled"]
        boot = block_bootstrap_diff(self.cells(cand, "validation"), self.cells(champion, "validation"))
        if d_val < 0:
            ok, why = False, f"rejected at validation: skill change {d_val:+.4f} (search had {d_search:+.4f})"
        elif d_guard < -GUARD_TOL:
            ok, why = False, f"rejected by guard: all-series skill change {d_guard:+.4f}"
        else:
            ok, why = True, (f"accepted: search {d_search:+.4f}, validation {d_val:+.4f} "
                             f"(bootstrap P(better) {boot.get('p_a_better', float('nan')):.2f}), guard {d_guard:+.4f}")
        entry = self._entry(iteration, hyp, champion, cand, s_c, s_n,
                            {"champion": v_c, "candidate": v_n, "bootstrap": boot},
                            {"champion": g_c, "candidate": g_n}, ok, why, t0)
        return ok, (cand if ok else champion), entry

    def _entry(self, iteration, hyp, champion, cand, s_c, s_n, val, guard, ok, why, t0) -> dict:
        return {
            "iteration": iteration,
            "hypothesis": {k: v for k, v in asdict(hyp).items()},
            "champion_config": champion.to_dict(),
            "candidate_config": cand.to_dict(),
            "search": {"champion": s_c, "candidate": s_n},
            "validation": val,
            "guard": guard,
            "accepted": ok,
            "decision": why,
            "seconds": round(time.time() - t0, 2),
        }

    def run_agenda(self, start: Config, agenda: list[Hypothesis], iteration0: int = 1,
                   log=print) -> tuple[Config, list[dict]]:
        champion = start
        entries = []
        for i, hyp in enumerate(agenda, start=iteration0):
            ok, champion, entry = self.test(champion, hyp, i)
            entries.append(entry)
            self._append(entry)
            log(f"[{i:02d}] {hyp.hid} {hyp.title}: {entry['decision']}")
        return champion, entries

    def _append(self, entry: dict) -> None:
        with open(self.nb_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=float) + "\n")
        h = entry["hypothesis"]
        s = entry["search"]
        lines = [f"### Iteration {entry['iteration']}: {h['hid']} {h['title']}",
                 "",
                 f"- **Claim:** {h['claim']}",
                 f"- **Pre-registered prediction:** {h['prediction']}",
                 f"- **Change tested:** `{json.dumps(h['delta'])}` (proposed by: {h['proposer']})"]
        if s.get("candidate"):
            lines.append(f"- **Search** pooled skill: champion {s['champion']['skill_pooled']:+.4f} -> "
                         f"candidate {s['candidate']['skill_pooled']:+.4f}; coverage "
                         f"{s['champion']['coverage_inside']:.2f} -> {s['candidate']['coverage_inside']:.2f}")
        v = entry["validation"]
        if v:
            lines.append(f"- **Validation** pooled skill: champion {v['champion']['skill_pooled']:+.4f} -> "
                         f"candidate {v['candidate']['skill_pooled']:+.4f}; bootstrap 90% CI of the gain "
                         f"[{v['bootstrap'].get('ci90', [float('nan')]*2)[0]:+.4f}, "
                         f"{v['bootstrap'].get('ci90', [float('nan')]*2)[1]:+.4f}]")
        g = entry["guard"]
        if g:
            lines.append(f"- **Guard** (all 592 rental series): {g['champion']['skill_pooled']:+.4f} -> "
                         f"{g['candidate']['skill_pooled']:+.4f}")
        lines.append(f"- **Decision:** {'ACCEPTED' if entry['accepted'] else 'rejected'}: {entry['decision']}")
        lines.append("")
        with open(self.nb_md, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
