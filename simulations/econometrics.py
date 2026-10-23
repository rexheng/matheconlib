"""Monte Carlo econometrics.

OLS sampling error, omitted-variable bias, instrumental variables, and a
sharp regression discontinuity. MathEconLib does not yet formalize
econometrics; these are numerical.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def _design(x: Array) -> Array:
    column = np.asarray(x, dtype=float).reshape(-1)
    return np.column_stack([np.ones(column.shape[0]), column])


def ols_slope(x: Array, y: Array) -> float:
    """Slope from an OLS regression of ``y`` on a constant and ``x``."""
    beta, *_ = np.linalg.lstsq(_design(x), np.asarray(y, dtype=float), rcond=None)
    return float(beta[1])


def tsls_slope(instrument: Array, endogenous: Array, outcome: Array) -> float:
    """Exactly identified 2SLS slope, with a constant in both stages."""
    instrument = np.asarray(instrument, dtype=float).reshape(-1)
    endogenous = np.asarray(endogenous, dtype=float).reshape(-1)
    outcome = np.asarray(outcome, dtype=float).reshape(-1)
    first, *_ = np.linalg.lstsq(_design(instrument), endogenous, rcond=None)
    fitted = _design(instrument) @ first
    return ols_slope(fitted, outcome)


def ols_sampling(
    intercept: float = 1.0,
    slope: float = 2.0,
    n: int = 200,
    reps: int = 800,
    noise: float = 1.0,
    seed: int = 0,
) -> Array:
    """Sampling distribution of the OLS slope in ``y = a + b x + e``."""
    rng = np.random.default_rng(seed)
    slopes = np.empty(reps, dtype=float)
    for i in range(reps):
        x = rng.normal(size=n)
        y = intercept + slope * x + rng.normal(scale=noise, size=n)
        slopes[i] = ols_slope(x, y)
    return slopes


def omitted_variable_bias(
    n: int = 500,
    reps: int = 400,
    seed: int = 1,
) -> Array:
    """Short-regression slopes when a relevant, correlated regressor is left out.

    The long model is ``y = x + 2 z + e`` with ``x = 0.8 z + u``. The
    population short slope on ``x`` exceeds 1 because ``z`` is omitted and
    moves with ``x``.
    """
    rng = np.random.default_rng(seed)
    slopes = np.empty(reps, dtype=float)
    for i in range(reps):
        omitted = rng.normal(size=n)
        x = 0.8 * omitted + rng.normal(scale=0.5, size=n)
        y = x + 2.0 * omitted + rng.normal(size=n)
        slopes[i] = ols_slope(x, y)
    return slopes


def population_omitted_slope() -> float:
    """Population short slope for the DGP in `omitted_variable_bias`.

    ``x = 0.8 z + u``, ``y = x + 2 z + e``, with ``z, u, e`` standard normal
    and independent, except ``u`` has variance ``0.25``. Then
    ``Cov(x, y) / Var(x) = (Var(x) + 1.6) / Var(x)``.
    """
    variance_x = 0.8**2 + 0.5**2
    covariance = variance_x + 2.0 * 0.8
    return float(covariance / variance_x)


def iv_versus_ols(
    n: int = 500,
    reps: int = 400,
    slope: float = 2.0,
    seed: int = 2,
) -> tuple[Array, Array]:
    """OLS and 2SLS slopes when the regressor shares a shock with the error.

    ``x = 0.7 z + u`` and ``y = β x + u + e``. OLS loads the common shock
    ``u`` onto the slope. The instrument ``z`` does not.
    """
    rng = np.random.default_rng(seed)
    ols = np.empty(reps, dtype=float)
    iv = np.empty(reps, dtype=float)
    for i in range(reps):
        instrument = rng.normal(size=n)
        shock = rng.normal(size=n)
        x = 0.7 * instrument + shock
        y = slope * x + shock + rng.normal(size=n)
        ols[i] = ols_slope(x, y)
        iv[i] = tsls_slope(instrument, x, y)
    return ols, iv


def rdd_local_linear(
    n: int = 2_000,
    effect: float = 1.5,
    cutoff: float = 0.0,
    bandwidth: float = 0.35,
    seed: int = 3,
) -> float:
    """Sharp-design effect from separate local linear fits on each side.

    The outcome is ``y = 0.3 + 0.8 r + τ 1{r >= c} + e`` with ``r`` uniform
    on ``[-1, 1]``. The reported number is the jump in the fitted lines at
    the cutoff, using observations inside the bandwidth.
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    rng = np.random.default_rng(seed)
    running = rng.uniform(-1.0, 1.0, size=n)
    outcome = (
        0.3
        + 0.8 * running
        + effect * (running >= cutoff)
        + rng.normal(scale=0.25, size=n)
    )
    left = (running >= cutoff - bandwidth) & (running < cutoff)
    right = (running >= cutoff) & (running <= cutoff + bandwidth)
    if left.sum() < 5 or right.sum() < 5:
        raise ValueError("bandwidth contains too few observations")
    left_fit = _local_line(running[left], outcome[left])
    right_fit = _local_line(running[right], outcome[right])
    return float((right_fit[0] + right_fit[1] * cutoff) - (left_fit[0] + left_fit[1] * cutoff))


def rdd_sample(
    n: int = 800, effect: float = 1.5, cutoff: float = 0.0, seed: int = 3
) -> tuple[Array, Array]:
    """One sharp-design sample ``(running, outcome)`` for plotting."""
    rng = np.random.default_rng(seed)
    running = rng.uniform(-1.0, 1.0, size=n)
    outcome = (
        0.3
        + 0.8 * running
        + effect * (running >= cutoff)
        + rng.normal(scale=0.25, size=n)
    )
    return running, outcome


def _local_line(x: Array, y: Array) -> Array:
    beta, *_ = np.linalg.lstsq(_design(x), y, rcond=None)
    return beta
