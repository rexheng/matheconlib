"""Numerical micro, macro, and econometrics playground.

The Lean library in this repository machine-checks microeconomic theory.
These modules are ordinary NumPy simulations. Where a result is also a
theorem, the docstring names it. Macro and econometrics sims are numerical
only: that material is in MathEconLib's long-term scope, not the current
formal development.
"""

from simulations.econometrics import (
    iv_versus_ols,
    omitted_variable_bias,
    ols_sampling,
    rdd_local_linear,
)
from simulations.macro import (
    golden_rule_saving,
    nk_impulse,
    solow_steady_state,
    solow_transition,
)
from simulations.micro import (
    becker_crime_rate,
    cournot_dynamics,
    cournot_nash,
    deferred_acceptance,
    fictitious_play_matching_pennies,
    is_stable,
    uniform_optimal_price,
    uniform_seller_profit,
)

__all__ = [
    "becker_crime_rate",
    "cournot_dynamics",
    "cournot_nash",
    "deferred_acceptance",
    "fictitious_play_matching_pennies",
    "golden_rule_saving",
    "is_stable",
    "iv_versus_ols",
    "nk_impulse",
    "ols_sampling",
    "omitted_variable_bias",
    "rdd_local_linear",
    "solow_steady_state",
    "solow_transition",
    "uniform_optimal_price",
    "uniform_seller_profit",
]
