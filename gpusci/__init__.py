"""gpusci: a small AI-scientist harness for short-horizon GPU rental price forecasts.

Modules:
    data       load the starter kit's series file and catalog into fast arrays
    model      the quantile forecaster (weighted empirical distribution of h-day log changes)
    score      pinball loss and skill, matching the starter kit's scorer
    backtest   rolling-origin evaluation over a universe of series
    scientist  the hypothesis -> test -> keep/reject -> log loop
"""
