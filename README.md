# gpusci: an AI scientist for short-horizon GPU rental prices

Entry for Track 2 (Pacific, "AI Scientist for GPU Value") of the Iterate AI Research Hack, 14-15 Nov 2026.
Work in progress.

## What it does

An AI agent (Claude) wrote a research agenda of 12 hypotheses about what moves posted GPU rental prices
(`gpusci/hypotheses.py`). Each one states an economic claim, a prediction written before testing, and the change
it makes to the forecaster. A fixed protocol (`gpusci/scientist.py`) then tests them one by one against the current
champion and keeps or rejects each. Every outcome is written to the lab notebook (`notebook/LAB_NOTEBOOK.md`),
rejections included.

**Forecaster** (`gpusci/model.py`): the 0.1/0.5/0.9 quantiles of today's price × exp(r), where r is drawn from a
weighted empirical distribution of past h-day log changes. Posted prices are sticky, so that distribution has a large
atom at zero. The hypotheses change which past changes count and how much each one weighs:
- recency weighting,
- pooling with peer series of the same instrument class,
- conditioning on market state,
- reverting after a provider joins or leaves.

**Protocol:**

| Set | Origins | Used for |
|---|---|---|
| search | 10 Aug – 5 Oct 2026 | screening each hypothesis |
| validation | 6 – 15 Oct 2026 | confirming a screened gain |
| locked exam | 23 – 30 Oct 2026 | scored once, after the loop ends; never used for a decision |

- No target dates overlap between the three sets.
- The scored universe is the 37 headline posted-rental series at the 6 Nov cutoff.
- A guard universe of all 592 posted-rental series stops a change from quietly hurting the sticky series.

## Results so far (pooled skill vs the flat last value; pinball loss at 0.1/0.5/0.9; horizons 1, 3, 7 days)

| model | search | validation | locked exam | exam, all 592 series |
|---|---|---|---|---|
| drift (starter kit) | +0.088 | +0.208 | +0.070 | +0.094 |
| champion (H02 + H03 + H05 + H12) | +0.163 | +0.259 | +0.109 | +0.181 |

- **Champion minus drift on the exam:** +0.039, 90% interval [+0.016, +0.065], resampling whole origins.
- **Caveat:** exam skill peaked after the third accepted change at +0.127. The last two accepted changes did not carry over to the exam, which suggests the 10-origin validation window is too small.

The full log is in `notebook/LAB_NOTEBOOK.md` and `results/exam_path.csv`.

## Reproduce

Requires Python 3.10+ with pandas, numpy, duckdb and pyarrow (`requirements.txt`), and the data pack
`gpu-price-pack-v0.1.1-handover`. The pack is not in this repository: it is about 300 MB and CC BY 4.0, distributed
by the organisers.

```bash
P=path/to/gpu-price-pack-v0.1.1-handover; O=out
python3 -B $P/starter/build_series.py --pack $P --out $O/series_daily.csv --catalog $O/series_catalog.csv --as-of 2026-11-06
python3 -B $P/starter/build_series.py --pack $P --cutoff 2026-11-06 --members-from $O/series_catalog.csv --out $O/history_2026-11-06.csv --catalog $O/catalog_2026-11-06.csv
python3 -B -m gpusci.cli loop --series $O/series_daily.csv --catalog $O/catalog_2026-11-06.csv
python3 -B -m gpusci.cli forecast --history $O/history_2026-11-06.csv --catalog $O/catalog_2026-11-06.csv --out submission/forecasts_2026-11-06.csv
```

The loop is deterministic and needs no network.

## Data and attribution

Data: Pacific Machines (2026), gpu-price-pack v0.1.1-handover (data cutoff 6 November 2026), CC BY 4.0 on the
compilation. Upstream sources are credited per the pack's `ATTRIBUTION.md`:
- GPU Price Tracker (A. Hubbard),
- gpurentalprices.com,
- FastGPU,
- Tanod,
- GetDeploying,
- AWS Spot Price History (E. Pauley, Zenodo),
- Epoch AI,
- MLCommons.

Every rental price in the pack is a posted price, not a price paid.
