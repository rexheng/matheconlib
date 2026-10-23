"""Run the gallery: ``python -m simulations``."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from simulations.econometrics import (
    iv_versus_ols,
    ols_sampling,
    omitted_variable_bias,
    population_omitted_slope,
    rdd_local_linear,
)
from simulations.macro import nk_impulse, solow_steady_state
from simulations.micro import (
    APPLICANTS,
    DEPARTMENTS,
    becker_crime_rate_uniform,
    cournot_nash,
    deferred_acceptance,
    fictitious_play_matching_pennies,
    uniform_optimal_price,
    uniform_seller_profit,
)
from simulations.plotting import write_gallery


def main() -> None:
    parser = argparse.ArgumentParser(description="MathEconLib numerical playground")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("simulations/output"),
        help="directory for PNG figures",
    )
    args = parser.parse_args()

    price = uniform_optimal_price(0.0, 1.0, 0.0)
    profit = uniform_seller_profit(price, 0.0, 1.0, 0.0)
    pennies = fictitious_play_matching_pennies(2_000, seed=0)
    applicant_match = deferred_acceptance(APPLICANTS, DEPARTMENTS)
    department_match = {
        person: dept
        for dept, person in deferred_acceptance(DEPARTMENTS, APPLICANTS).items()
    }
    monetary = nk_impulse(shock="monetary")
    demand = nk_impulse(shock="demand")
    ols = ols_sampling(reps=400)
    short = omitted_variable_bias(reps=200)
    naive, instrumented = iv_versus_ols(reps=200)
    jump = rdd_local_linear()

    print("Micro")
    print(f"  uniform monopoly price on [0, 1]: {price:.2f}, profit {profit:.2f}")
    print(f"  Becker crime rate at p=0.5, f=1: {becker_crime_rate_uniform(0.5, 1.0):.2f}")
    print(
        "  matching-pennies heads frequency, last 200 rounds: "
        f"{pennies[-200:].mean(axis=0).round(3).tolist()}"
    )
    print(f"  Cournot Nash quantity per firm: {cournot_nash(10.0, 1.0, 1.0):.3f}")
    print(f"  applicant-proposing match: {applicant_match}")
    print(f"  department-proposing match: {department_match}")
    print("Macro")
    print(f"  Solow k* at s=0.3, delta=0.08, alpha=0.35: {solow_steady_state(0.3, 0.08, 0.35):.3f}")
    print(
        "  NK impact, monetary (output, inflation): "
        f"({monetary['output'][0]:+.3f}, {monetary['inflation'][0]:+.3f})"
    )
    print(
        "  NK impact, demand (output, inflation): "
        f"({demand['output'][0]:+.3f}, {demand['inflation'][0]:+.3f})"
    )
    print("Econometrics")
    print(f"  OLS mean slope (truth 2): {float(np.mean(ols)):.3f}")
    print(
        "  omitted-variable mean slope "
        f"(population {population_omitted_slope():.3f}): {float(np.mean(short)):.3f}"
    )
    print(
        "  endogenous regressor, OLS vs 2SLS means: "
        f"{float(np.mean(naive)):.3f} vs {float(np.mean(instrumented)):.3f}"
    )
    print(f"  sharp RDD jump (truth 1.5): {jump:.3f}")

    paths = write_gallery(args.out)
    print("Figures")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
