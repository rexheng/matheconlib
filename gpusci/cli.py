"""Command line: `python -m gpusci.cli loop|forecast ...` (see reproduce.sh for the full pipeline)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from .backtest import eligible
from .data import headline_ids, load_store, to_day, to_iso
from .hypotheses import AGENDA
from .model import Config, PeerPools, forecast
from .scientist import Lab, PROTOCOL
from .score import block_bootstrap_diff, summarize

FLAT = Config(flat_classes=("hyperscaler_list", "hyperscaler_spot", "neocloud_rate_card", "neocloud_spot",
                            "marketplace_ask", "aggregator_cheapest"))


def md_table(df: pd.DataFrame) -> str:
    def fmt(v):
        return f"{v:+.4f}" if isinstance(v, float) else str(v)
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False))


def cmd_loop(a) -> None:
    os.makedirs(a.notebook_dir, exist_ok=True)
    os.makedirs(a.results_dir, exist_ok=True)
    nb_jsonl = os.path.join(a.notebook_dir, "lab_notebook.jsonl")
    nb_md = os.path.join(a.notebook_dir, "LAB_NOTEBOOK.md")
    for p in (nb_jsonl, nb_md):
        if os.path.exists(p):
            os.remove(p)
    store = load_store(a.series, a.catalog)
    scored = headline_ids(a.catalog)
    guard = sorted(store)
    lab = Lab(store, scored, guard, nb_jsonl, nb_md)
    start = Config()
    s0, v0 = lab.metrics(start, "search"), lab.metrics(start, "validation")
    with open(nb_md, "w", encoding="utf-8") as f:
        f.write("# Lab notebook\n\n"
                "Written by the loop (`python -m gpusci.cli loop`). Every hypothesis is logged with its claim, "
                "its pre-registered prediction, the change tested and the decision, accepted or not.\n\n"
                f"**Protocol.** Search origins {PROTOCOL['search'][0]}..{PROTOCOL['search'][1]}, validation "
                f"{PROTOCOL['validation'][0]}..{PROTOCOL['validation'][1]}, locked exam "
                f"{PROTOCOL['exam'][0]}..{PROTOCOL['exam'][1]} (never used for a decision). Horizons 1, 3, 7 days. "
                f"Accept when search skill gains at least {PROTOCOL['min_gain']}, validation does not fall and the "
                f"all-series guard does not fall by more than {PROTOCOL['guard_tol']}.\n\n"
                f"**Scored universe.** {len(scored)} headline posted-rental series of the catalog at the event cutoff. "
                f"**Guard universe.** {len(guard)} posted-rental series.\n\n"
                f"**Starting point:** the starter kit's drift baseline. Search skill {s0['skill_pooled']:+.4f} "
                f"(coverage {s0['coverage_inside']:.2f}), validation {v0['skill_pooled']:+.4f}.\n\n")
    champion, entries = lab.run_agenda(start, AGENDA)
    path = [("start", start)] + [(e["hypothesis"]["hid"], Config.from_dict(e["candidate_config"]))
                                 for e in entries if e["accepted"]]

    # ---- the locked exam: scored once, after every decision has been made -------------------
    rows = []
    baselines = [("last_value (flat)", FLAT), ("drift (starter)", Config())]
    for name, cfg in baselines + [(f"champion after {hid}", cfg) for hid, cfg in path]:
        ex, eg = lab.metrics(cfg, "exam"), lab.metrics(cfg, "exam_guard")
        se, va = lab.metrics(cfg, "search"), lab.metrics(cfg, "validation")
        rows.append({"model": name, "search_skill": se["skill_pooled"], "validation_skill": va["skill_pooled"],
                     "exam_skill": ex["skill_pooled"], "exam_skill_h1": ex.get("skill_pooled_h1"),
                     "exam_skill_h3": ex.get("skill_pooled_h3"), "exam_skill_h7": ex.get("skill_pooled_h7"),
                     "exam_median_of_series": ex["skill_median_of_series"], "exam_geomean": ex["skill_geomean"],
                     "exam_coverage": ex["coverage_inside"], "exam_guard_skill": eg["skill_pooled"],
                     "config": json.dumps(cfg.to_dict())})
    exam = pd.DataFrame(rows)
    exam.to_csv(os.path.join(a.results_dir, "exam_path.csv"), index=False)
    boot = block_bootstrap_diff(lab.cells(champion, "exam"), lab.cells(Config(), "exam"))
    lab.cells(champion, "exam").to_csv(os.path.join(a.results_dir, "exam_cells_champion.csv"), index=False)
    lab.cells(Config(), "exam").to_csv(os.path.join(a.results_dir, "exam_cells_drift.csv"), index=False)
    with open(os.path.join(a.results_dir, "champion.json"), "w", encoding="utf-8") as f:
        json.dump({"config": champion.to_dict(), "path": [h for h, _ in path],
                   "exam_vs_drift_bootstrap": boot, "protocol": PROTOCOL}, f, indent=1)
    with open(nb_md, "a", encoding="utf-8") as f:
        f.write("## Locked exam (scored once, after the loop)\n\n")
        f.write(md_table(exam.drop(columns=["config"])) + "\n\n")
        f.write(f"Champion minus drift on the exam, pooled skill: {boot['skill_diff']:+.4f}, 90% interval "
                f"[{boot['ci90'][0]:+.4f}, {boot['ci90'][1]:+.4f}] by resampling whole origins "
                f"({boot['origins']} origins); P(champion better) = {boot['p_a_better']:.2f}.\n")
    print(exam.drop(columns=["config"]).to_string(index=False))
    print("exam bootstrap champion vs drift:", boot)


def cmd_forecast(a) -> None:
    store = load_store(a.history, a.catalog)
    t = to_day(a.origin)
    late = [s for s in store.values() if len(s.days) and s.days[-1] > t]
    if late:
        sys.exit(f"REFUSED: history has rows after the origin {a.origin} ({late[0].sid})")
    cfg = Config.from_dict(json.load(open(a.champion))["config"])
    pools = PeerPools(store, cfg.peer_group) if cfg.peer_group else None
    ids = headline_ids(a.catalog) if a.ids == "headline" else sorted(store)
    rows = []
    for sid in ids:
        s = store[sid]
        if not eligible(s, t):
            continue
        for h in (1, 3, 7):
            q10, q50, q90, last, note = forecast(s, t, h, cfg, pools)
            rows.append({"series_id": sid, "origin_date": a.origin, "horizon_days": h,
                         "q10": round(q10, 6), "q50": round(q50, 6), "q90": round(q90, 6),
                         "target_date": to_iso(t + h), "team_id": a.team_id,
                         "method_note": f"gpusci champion ({note})"})
    out = pd.DataFrame(rows)
    assert (out.q10 > 0).all() and (out.q10 <= out.q50).all() and (out.q50 <= out.q90).all()
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(out)} rows, {out.series_id.nunique()} series, origin {a.origin}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="gpusci")
    sub = p.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("loop", help="run the scientist loop and the locked exam")
    l.add_argument("--series", required=True, help="series_daily.csv from build_series.py (as-of the cutoff)")
    l.add_argument("--catalog", required=True, help="catalog at the event cutoff")
    l.add_argument("--notebook-dir", default="notebook")
    l.add_argument("--results-dir", default="results")
    l.set_defaults(fn=cmd_loop)
    f = sub.add_parser("forecast", help="write the submission from the champion")
    f.add_argument("--history", required=True, help="history_<cutoff>.csv (truncated at the origin)")
    f.add_argument("--catalog", required=True)
    f.add_argument("--champion", default="results/champion.json")
    f.add_argument("--origin", default="2026-11-06")
    f.add_argument("--ids", choices=["all", "headline"], default="all")
    f.add_argument("--team-id", default="gpusci")
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_forecast)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
