"""Small macroeconomic simulations.

A Solow growth model and a three-equation New Keynesian linearization.
MathEconLib does not yet formalize macroeconomics; these are numerical.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def solow_steady_state(
    saving: float, depreciation: float, alpha: float, productivity: float = 1.0
) -> float:
    """Steady-state capital in ``k' = s A k^α + (1 - δ) k``.

    There is no population growth and no technical progress, so
    ``k* = (s A / δ) ** (1 / (1 - α))``.
    """
    _check_solow(saving, depreciation, alpha, productivity)
    return float((saving * productivity / depreciation) ** (1.0 / (1.0 - alpha)))


def solow_transition(
    capital0: float,
    saving: float,
    depreciation: float,
    alpha: float,
    productivity: float = 1.0,
    periods: int = 80,
) -> Array:
    """Capital path of length ``periods``, including the initial condition."""
    _check_solow(saving, depreciation, alpha, productivity)
    if capital0 <= 0:
        raise ValueError("initial capital must be positive")
    if periods < 1:
        raise ValueError("periods must be positive")
    capital = np.empty(periods, dtype=float)
    capital[0] = capital0
    for t in range(periods - 1):
        output = productivity * capital[t] ** alpha
        capital[t + 1] = saving * output + (1.0 - depreciation) * capital[t]
    return capital


def steady_consumption(
    saving: float, depreciation: float, alpha: float, productivity: float = 1.0
) -> float:
    """Steady-state consumption ``(1 - s) A k*^α``."""
    capital = solow_steady_state(saving, depreciation, alpha, productivity)
    output = productivity * capital ** alpha
    return float((1.0 - saving) * output)


def golden_rule_saving(alpha: float) -> float:
    """Saving rate that maximizes steady-state consumption.

    With only depreciation eating capital, that rate equals capital's
    income share ``α``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return float(alpha)


def nk_loadings(
    beta: float,
    sigma: float,
    kappa: float,
    phi_pi: float,
    phi_x: float,
    rho: float,
    shock: str,
) -> tuple[float, float]:
    """Impact loadings ``(output gap, inflation)`` on a unit AR(1) shock.

    The system is

    - IS: ``x = E x' - σ (i - E π' - rⁿ)``
    - Phillips: ``π = β E π' + κ x``
    - Taylor: ``i = φ_π π + φ_x x + v``

    ``shock="monetary"`` is a unit innovation to ``v``, with ``rⁿ = 0``.
    ``shock="demand"`` is a unit innovation to ``rⁿ``, with ``v = 0``.
    Both shocks persist at rate ``rho``. Guess ``x = a ε``, ``π = b ε``.
    """
    if shock not in {"monetary", "demand"}:
        raise ValueError("shock must be 'monetary' or 'demand'")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie in (0, 1)")
    if sigma <= 0 or kappa <= 0:
        raise ValueError("sigma and kappa must be positive")
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must lie in (-1, 1)")
    factor = (1.0 - beta * rho) * (1.0 - rho + sigma * phi_x) / kappa + sigma * (
        phi_pi - rho
    )
    if abs(factor) < 1e-12:
        raise ValueError("equilibrium coefficient is singular at these parameters")
    inflation = (-sigma if shock == "monetary" else sigma) / factor
    output = inflation * (1.0 - beta * rho) / kappa
    return float(output), float(inflation)


def nk_impulse(
    beta: float = 0.99,
    sigma: float = 1.0,
    kappa: float = 0.15,
    phi_pi: float = 1.5,
    phi_x: float = 0.125,
    rho: float = 0.6,
    shock: str = "monetary",
    horizon: int = 24,
    scale: float = 1.0,
) -> dict[str, Array]:
    """Impulse responses. Period 0 is the impact of a shock of size ``scale``."""
    output0, inflation0 = nk_loadings(beta, sigma, kappa, phi_pi, phi_x, rho, shock)
    decay = rho ** np.arange(horizon, dtype=float)
    output = scale * output0 * decay
    inflation = scale * inflation0 * decay
    if shock == "monetary":
        natural = np.zeros(horizon)
        policy = scale * decay
    else:
        natural = scale * decay
        policy = np.zeros(horizon)
    nominal = phi_pi * inflation + phi_x * output + policy
    return {
        "output": output,
        "inflation": inflation,
        "nominal_rate": nominal,
        "natural_rate": natural,
        "policy_shock": policy,
    }


def nk_residuals(
    beta: float,
    sigma: float,
    kappa: float,
    phi_pi: float,
    phi_x: float,
    rho: float,
    shock: str,
) -> tuple[float, float]:
    """IS and Phillips residuals on impact, which a solution must drive to zero."""
    output, inflation = nk_loadings(beta, sigma, kappa, phi_pi, phi_x, rho, shock)
    expected_output = rho * output
    expected_inflation = rho * inflation
    policy = 1.0 if shock == "monetary" else 0.0
    natural = 0.0 if shock == "monetary" else 1.0
    nominal = phi_pi * inflation + phi_x * output + policy
    is_gap = output - (expected_output - sigma * (nominal - expected_inflation - natural))
    pc_gap = inflation - (beta * expected_inflation + kappa * output)
    return float(is_gap), float(pc_gap)


def _check_solow(saving: float, depreciation: float, alpha: float, productivity: float) -> None:
    if not 0.0 < saving < 1.0:
        raise ValueError("saving rate must lie in (0, 1)")
    if not 0.0 < depreciation < 1.0:
        raise ValueError("depreciation must lie in (0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if productivity <= 0:
        raise ValueError("productivity must be positive")
